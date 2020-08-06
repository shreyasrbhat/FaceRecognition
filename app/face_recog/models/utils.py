import matplotlib.pyplot as plt
import cv2

def display_sample(main_img, test_set, label):
    fig = plt.figure(figsize=(25,4))
    test_set = test_set.squeeze(0)
    num_test = test_set.shape[0]

    ax1 = fig.add_subplot(2,num_test,label.item()+1, xticks=[], yticks=[])
    img = main_img.numpy().squeeze(0).squeeze(0)
    plt.subplots_adjust(wspace=0, hspace=0)
    ax1.imshow(img)
    for i in range(num_test):

        ax2 = fig.add_subplot(2, num_test, i+1+num_test, xticks=[], yticks=[])
        img = test_set[i].numpy().squeeze(0)
        plt.subplots_adjust(wspace=0, hspace=0)
        ax2.imshow(img)

def plot_face(face_tensor):
    b,g,r = cv2.split(face_tensor.permute(1,2,0).numpy())
    plt.imshow(cv2.merge([r,g,b]))
    plt.axis('off') 