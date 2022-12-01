if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import seaborn as sns

    sns.set(rc={'figure.figsize': (50, 20)})

    # sns.set(color_codes=True)
    # mpl.rcParams["font.sans-serif"] = ["SimHei"]
    # mpl.rcParams["axes.unicode_minus"] = False

    # 柱高信息
    Y = [0.8544, 0.7627, 0.8038, 0.8639, 0.9146, 0.9335, 0.9304, 0.9587, 0.9810,
            0.6171, 0.5048, 0.3111, 0.8291, 0.9367, 0.9494, 0.8762, 0.7753, 0.8734,
            0.9082, 0.9365, 0.9715, 0.9525, 0.8291, 0.9684, 0.8892, 0.9937, 0.9937,
            0.8576, 0.7595, 0.6772, 0.7873, 0.7848, 0.8829, 0.8228, 0.9335, 0.9652,
            0.8228, 0.8703, 0.8671, 0.8558, 0.8449, 0.9778, 1.0000, 0.7373, 0.8734,
            0.7373, 0.8544, 0.9367, 0.8481, 0.8882, 0.9363, 0.9302, 0.7943, 0.8662,
            0.9159, 0.8956, 0.9177, 0.9620, 0.9430, 0.9617]
    Y1 = [0.7816, 0.7437, 0.8101, 0.6804, 0.8829, 0.8861, 0.8418, 0.9619, 0.9589,
            0.5506, 0.5016, 0.2889, 0.7563, 0.9494, 0.9146, 0.7238, 0.7753, 0.8133,
            0.9082, 0.8889, 0.9430, 0.8861, 0.7595, 0.9399, 0.7880, 0.9810, 0.9937,
            0.7215, 0.7405, 0.5570, 0.7143, 0.5981, 0.8418, 0.7468, 0.9272, 0.8987,
            0.7753, 0.8101, 0.7816, 0.7981, 0.8006, 0.9462, 0.9968, 0.6741, 0.7690,
            0.6899, 0.8133, 0.9019, 0.8259, 0.8019, 0.9140, 0.8984, 0.7089, 0.8726,
            0.8544, 0.8703, 0.8829, 0.9430, 0.9589, 0.9681]
    Y2 = [0.7880, 0.7405, 0.7785, 0.8576, 0.8544, 0.8703, 0.9082, 0.9587, 0.9778,
        0.5949, 0.4063, 0.3810, 0.6867, 0.9494, 0.9177, 0.8032, 0.8101, 0.7690,
        0.8481, 0.8254, 0.8987, 0.9082, 0.8133, 0.9177, 0.7722, 0.9905, 0.9842,
        0.7848, 0.7025, 0.6772, 0.7810, 0.7278, 0.8006, 0.6329, 0.9494, 0.8987,
        0.6962, 0.8133, 0.7785, 0.8333, 0.7943, 0.9209, 0.7848, 0.6013, 0.7342,
        0.7310, 0.7563, 0.8196, 0.7342, 0.8658, 0.9076, 0.8603, 0.7722, 0.8662,
        0.8220, 0.8513, 0.8070, 0.9019, 0.9462, 0.8978]
    Y3 = [0.8196, 0.7152, 0.7057, 0.8418 , 0.8133, 0.8829, 0.9241, 0.9492, 0.9715,
        0.5000, 0.3746, 0.3587, 0.6930, 0.9399, 0.9082, 0.7905, 0.7215, 0.8449,
        0.8481, 0.9492, 0.9557, 0.9525, 0.8259, 0.9209, 0.7975, 0.9905, 0.9937,
        0.7627, 0.6994, 0.6677, 0.7143, 0.6930, 0.7785, 0.7120, 0.9177, 0.9494,
        0.7848, 0.9051, 0.8291, 0.8750, 0.8259, 0.9430, 0.9937, 0.8101, 0.7848,
        0.7468, 0.7848, 0.9146, 0.8133, 0.8722, 0.9331, 0.9143, 0.6772, 0.8822,
        0.8964, 0.8513, 0.7880, 0.9051, 0.9462, 0.9585]

    Y = Y[45:]
    Y1 = Y1[45:]
    Y2 = Y2[45:]
    Y3 = Y3[45:]

    X = np.arange(len(Y)) + 1

    bar_width = 0.20

    # tick_label = []
    # for i in range(60):
    #     tick_label.append(str(i + 1))

    # 显示每个柱的具体高度
    # for x, y in zip(X, Y):
    #     plt.text(x + 0.005, y + 0.005, '%.3f' % y, ha='center', va='bottom')
    #
    # for x, y1 in zip(X, Y1):
    #     plt.text(x + 0.24, y1 + 0.005, '%.3f' % y1, ha='center', va='bottom')

    # 绘制柱状图
    plt.bar(X, Y, bar_width, align="center", color="#dd8757", label="initial_stgcn", alpha=0.5)
    plt.bar(X + bar_width, Y1, bar_width, color="#587BB5", align="center", label="add_3d_module", alpha=0.5, )
    plt.bar(X + bar_width*2, Y2, bar_width, align="center", color="black", label="add_3d_module_poolT75_25", alpha=0.5)
    plt.bar(X + bar_width*3, Y3, bar_width, align="center", color="green", label="add_3d_module_poolT75_50", alpha=0.5)


    # plt.ylabel('The number of steps required to reach convergence', fontdict={'family': 'Times New Roman', 'size': 18})
    # plt.xlabel(' Number of experimental group', fontdict={'family': 'Times New Roman', 'size': 18})
    # plt.xlabel('Experimental no')
    # plt.ylabel('The number of convergence iterations')
    # plt.title('Picture Name')

    # plt.xticks(X + bar_width / 2, tick_label)
    # 显示图例
    plt.legend()
    # plt.legend(prop={'family': 'Times New Roman', 'size': 12})
    plt.show()
    # plt.savefig('result.jpg')
