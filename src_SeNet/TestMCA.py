class_weights = [0.036, 0.002, 0.084, 0.134, 0.037, 0.391, 0.316]

def getMCA(correct, predicted):
    mca = 0
    for lbl, w in enumerate(class_weights):
        count = 0.0
        tot = 0.0
        for i, x in enumerate(correct):
            if x == lbl:
                tot = tot + 1
                if x == predicted[i]:
                    count = count + 1

        acc_t = count / tot * 100.0
        mca = mca + acc_t
    mca = mca / len(class_weights)

    acc = 0
    for i, x in enumerate(correct):
        if x == predicted[i]:
            acc = acc + 1

    acc = acc / len(predicted) * 100
    return acc, mca

correct = [0, 1, 1, 2, 3, 4, 5, 6]
predict = [0, 2, 1, 2, 3, 4, 5, 6]
print(getMCA(correct, predict))