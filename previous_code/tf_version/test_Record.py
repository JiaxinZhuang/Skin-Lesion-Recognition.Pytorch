
import data_utils
import tensorflow as tf
import timer

def load(iter_x, iter_y, init):
    count = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(init)
        try:
            while True:
                _, y = sess.run([iter_x, iter_y])
                count += y.shape[0]
        except tf.errors.OutOfRangeError:
            print('Count is %d' % count)

def test_record():
    timer_ = timer.timer()
    ISIC2018_data_ =  data_utils.ISIC2018_data()
    ISIC2018_data_.set_valid_index(4)
    iter_x, iter_y = ISIC2018_data_.get_inputs()
    load(iter_x, iter_y, ISIC2018_data_.extra_init)
    #load(*(ISIC2018_data_.get_train))
    #load(*(ISIC2018_data_.get_valid))
    timer_.get_duration()

if __name__=='__main__':
    test_record()
