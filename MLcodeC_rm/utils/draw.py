import matplotlib.pyplot as plt

def draw(name, trainl, testl, type='Loss'):

    plt.title(name)  # 命名
    plt.plot(trainl, c='g', label='Train ' + type)
    plt.plot(testl, c='r', label='Test ' + type)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()