import numpy as np

def make_batchs(batch_size, train_dataset, train_labels, test_dataset ,test_labels):
    train_dataset_list = list()
    train_labels_list = list()

    test_dataset_list = list()
    test_labels_list = list()

    train_batches_per_epoch = len(train_dataset)//batch_size +1
    val_batches_per_epoch = len(test_dataset)//batch_size +1

    indices = np.random.permutation(len(train_dataset))
    train_dataset = train_dataset[indices]
    train_labels = train_labels[indices]

    for i in range(train_batches_per_epoch):
        train_dataset_list.append(train_dataset[i*batch_size:(i+1)*batch_size])
        train_labels_list.append(train_labels[i*batch_size:(i+1)*batch_size])
        if i==len(train_dataset):
            train_dataset_list.append(train_dataset[i*batch_size:])
            train_labels_list.append(train_labels[i*batch_size:])

    for i in range(val_batches_per_epoch):
        test_dataset_list.append(test_dataset[i*batch_size:(i+1)*batch_size])
        test_labels_list.append(test_labels[i*batch_size:(i+1)*batch_size])
        if i==len(test_dataset):
            test_dataset_list.append(test_dataset[i*batch_size:])
            test_labels_list.append(test_labels[i*batch_size:])
            
    return train_dataset_list, train_labels_list, test_dataset_list, test_labels_list, train_batches_per_epoch, val_batches_per_epoch