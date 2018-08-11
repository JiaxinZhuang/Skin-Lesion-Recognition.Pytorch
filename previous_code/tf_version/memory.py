import tensorflow as tf
#import numpy as np
#import logging

class Memory():
    """Memory Module"""

    def __init__(self, hps, key_dim=192, memory_size=8192, choose_k=1, alpha=0.1, correct_in_top=1, age_noise=8.0, threshold=0.5):
        self.key_dim = key_dim
        self.memory_size = memory_size
        self.choose_k = min(choose_k, memory_size)
        self.correct_in_top = correct_in_top
        self.age_noise = age_noise
        self.threshold = tf.get_variable('threshold', [1], dtype=tf.float32, trainable=False,
                initializer=tf.constant_initializer(0.8, tf.float32))
        #self.num_keys = tf.get_variable('nums_key', [1], dtype=tf.int32, trainable=False,
        #        initializer=tf.constant_initializer(0, tf.int32))

        self.mem_keys = tf.get_variable(
                'memkeys', [self.memory_size, self.key_dim], trainable=False,
                initializer=tf.random_uniform_initializer(-1.0, -1.0))
        self.mem_vals = tf.get_variable(
                'memvals', [self.memory_size, hps.num_classes], dtype=tf.float32, trainable=False,
                initializer=tf.constant_initializer(-1, tf.float32))
        self.mem_age = tf.get_variable(
                'memage', [self.memory_size], dtype=tf.float32, trainable=False,
                initializer=tf.constant_initializer(0, tf.float32))
        self.query_proj = tf.get_variable(
                'memory_query_proj', [self.key_dim, self.key_dim], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(0, 0.01))


    #def clear(self):
    #    return tf.variables_initializer([self.mem_keys, self.mem_vals, self.mem_age, self.recent_idx])


    def get_hint_pool_idxs(self, normalized_query):

        #num_keys = tf.identity(self.num_keys)
        #num_keys = tf.squeeze(num_keys)

        #part_mem_keys = tf.gather(self.mem_keys, tf.range(num_keys))


        similarities = tf.matmul(tf.stop_gradient(normalized_query),
                                 self.mem_keys, transpose_b=True, name='nn_mmul')

        hint_pool_vals, hint_pool_idxs = tf.nn.top_k(
                tf.stop_gradient(similarities), k=self.choose_k, name='nn_topl')

        return hint_pool_vals, hint_pool_idxs


    def query(self, query_vec):
        """Query memory

        Returns:
            [batch_size, labels]
        """

        batch_size = tf.shape(query_vec)[0]
        #query_vec = tf.matmul(query_vec, self.query_proj)
        normalized_query = tf.nn.l2_normalize(query_vec, dim=1)

        # get top
        _, hint_pool_idxs = self.get_hint_pool_idxs(normalized_query)

        #result = tf.gather(self.mem_vals, tf.reshape(hint_pool_idxs, [batch_size, -1]))
        # reshape is wierd here, just for k==1
        result = tf.gather(self.mem_vals, tf.reshape(hint_pool_idxs, [batch_size]))
        #result = tf.identity(result)
        return result


    #def make_update_op(self, upd_idxs, upd_keys, upd_vals,
    #                   batch_size, use_recent_idx, intended_output):
    #    mem_age_incr = self.mem_age.assign_add(tf.ones([self.memory_size],
    #                                           dtype=tf.float32))

    #    with tf.control_dependencies([mem_age_incr]):
    #        mem_age_upd = tf.scatter_update(
    #                self.mem_age, upd_idxs, tf.zeros([batch_size], dtype=tf.float32))

    #    mem_key_upd = tf.scatter_update(
    #            self.mem_keys, upd_idxs, upd_keys)
    #    mem_val_upd = tf.scatter_update(
    #            self.mem_vals, upd_idxs, upd_vals)

    #    if use_recent_idx:
    #        recent_idx_upd = tf.scatter_update(
    #                self.recent_idx, intended_output, upd_idxs)
    #    else:
    #        recent_idx_upd = tf.group()

    #    return tf.group(mem_age_upd, mem_key_upd, mem_val_upd, recent_idx_upd)


    def query_and_make_update(self, query_vec, intended_output):
        """receive query_vec and intended_output to udpate memory

            return:
                loss

            1. normalize query_vec
            2. get hint_pool_idxs for query_vec
                2.1 No Return: insert directly
                2.2 Return the indexs, and its sims
            3. compare sims with threshold
                3.1 sims bigger than threshold: no matter what label it is, just insert a new key value
                3.2 sims smaller than threshold
                    3.2.1 same label: update by query + key -> key
                    3.2.2 different label: insert a new key value
        """

        # Normalize for query_vec
        batch_size = tf.shape(query_vec)[0]
        normalized_query = tf.nn.l2_normalize(query_vec, dim=1)

        sims, hint_pool_idxs = self.get_hint_pool_idxs(normalized_query)

        #print(intended_output.get_shape)
        # incorrect_memory_lookup
        more_than_threshold = tf.less(self.threshold, sims)
        more_than_threshold = tf.reshape(more_than_threshold, [batch_size])
        # hint_pool_idxs (?,1) -> (?)
        hint_pool_idxs = tf.reshape(hint_pool_idxs, [batch_size])
        predictions = tf.gather(self.mem_vals, hint_pool_idxs, axis=0)
        #predictions = tf.reshape(predictions, [batch_size, -1])
        print(hint_pool_idxs)
        print(self.mem_vals)
        print(predictions)
        print(intended_output)
        mask = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(intended_output, axis=1))
        mask = tf.cast(mask, tf.int32)
        more_than_threshold = tf.cast(more_than_threshold, tf.int32)
        print(mask)
        print(more_than_threshold)
        incorrect_memory_lookup = mask * more_than_threshold
        incorrect_memory_lookup = tf.cast(incorrect_memory_lookup, tf.bool)
        print('here')
        print(incorrect_memory_lookup)


        # update indexs
        _, oldest_idxs = tf.nn.top_k(self.mem_age, k=batch_size, sorted=False)
        fetched_idxs = tf.reshape(hint_pool_idxs, [-1])

        # keys
        update_keys = normalized_query
        fetched_keys = tf.gather(self.mem_keys, fetched_idxs, name='fetched_keys')
        fetched_keys_upd = update_keys + fetched_keys  # Momentum-like update
        fetched_keys_upd = tf.nn.l2_normalize(fetched_keys_upd, dim=1)

        # vals
        fetched_vals = tf.gather(self.mem_vals, fetched_idxs, name='fetched_vals')
        update_vals = intended_output

        #with tf.control_dependencies([result]):
        upd_idxs = tf.where(incorrect_memory_lookup,
                      fetched_idxs,
                      oldest_idxs)
        upd_keys = tf.where(incorrect_memory_lookup,
                      fetched_keys_upd,
                      update_keys)
        upd_vals = tf.where(incorrect_memory_lookup,
                      fetched_vals,
                      update_vals)

        def make_update_op():
            return self.make_update_op(upd_idxs, upd_keys, upd_vals, batch_size)

        update_op = make_update_op()

        result = update_vals
        with tf.control_dependencies([update_op]):
            result = tf.identity(result)
            #loss = tf.identity(loss)

        return result


    def make_update_op(self, upd_idxs, upd_keys, upd_vals, batch_size):
        mem_age_incr = self.mem_age.assign_add(tf.ones([self.memory_size],
                                                   dtype=tf.float32))
        with tf.control_dependencies([mem_age_incr]):
            mem_age_upd = tf.scatter_update(
                self.mem_age, upd_idxs, tf.zeros([batch_size], dtype=tf.float32))

        mem_key_upd = tf.scatter_update(
            self.mem_keys, upd_idxs, upd_keys)
        mem_val_upd = tf.scatter_update(
            self.mem_vals, upd_idxs, upd_vals)

        return tf.group(mem_age_upd, mem_key_upd, mem_val_upd)
