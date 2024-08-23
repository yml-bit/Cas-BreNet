import numpy as np
import matplotlib.pyplot as plt
import os

def conf_index(confusion_matrix):
    # 2-TP/TN/FP/FN的计算
    weight=confusion_matrix.sum(axis=0)/confusion_matrix.sum()## 求出每列元素的和
    FN = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FP = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)#所有对的 TP.sum=TP+TN
    TN = confusion_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # print(TP)
    # print(TN)
    # print(FP)
    # print(FN)

    # 3-其他的性能参数的计算
    TPR = TP / (TP + FN)  # Sensitivity/ hit rate/ recall/ true positive rate 对于ground truth
    TNR = TN / (TN + FP)  # Specificity/ true negative rate  对于
    PPV = TP / (TP + FP)  # Precision/ positive predictive value  对于预测而言
    NPV = TN / (TN + FN)  # Negative predictive value
    FPR = FP / (FP + TN)  # Fall out/ false positive rate
    FNR = FN / (TP + FN)  # False negative rate
    FDR = FP / (TP + FP)  # False discovery rate
    sub_ACC = TP / (TP + FN)  # accuracy of each class
    acc=(TP+TN).sum()/(TP+TN+FP+FN).sum()
    average_acc=TP.sum() / (TP.sum() + FN.sum())
    F1_Score=2*TPR*PPV/(PPV+TPR)
    Macro_F1=F1_Score.mean()
    weight_F1=(F1_Score*weight).sum()# 应该把不同类别给与相同权重，不应该按照数量进行加权把？
    print('acc:',average_acc)
    print('Sensitivity:', TPR.mean())#Macro-average方法
    print('Specificity:', TNR.mean())
    print('Macro_F1:',Macro_F1)

def confuse_plot(cm, save_path):
    save_path += ".tif"
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Oranges')

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('数量', rotation=-90, va="bottom", fontsize=18)  # 设置颜色条标题的字号

    # 调整颜色条的刻度标签字体大小
    cbar.ax.tick_params(labelsize=18)

    # 显示数值
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > (cm.max() / 2.) else "black",
                    fontsize=18)  # 设置数值的字号

    # 设置坐标轴标签
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xticklabels(['Luminal', 'Non-Luminal', 'Benign'], fontsize=18)  # 设置X轴标签字号
    ax.set_yticklabels(['Luminal', 'Non-Luminal', 'Benign'], fontsize=18)  # 设置Y轴标签字号

    # 旋转顶部的标签，避免重叠
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", fontsize=18)  # 设置X轴刻度字号

    # 设定底部和右侧的边框不可见
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # 设定底部和左侧的边框线宽
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # 调整子图布局，防止坐标标签被截断
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    # plt.show()
    plt.close(fig)

def ca1():
    matrixr1a = np.array(
        [[54,14,36],
         [11,12,8],
        [2,0,47]])
    # conf_index(matrix)

    # 1-混淆矩阵
    matrixr2a = np.array(
        [[81,20,3],
          [17,12,2],
         [13,3,33]])

    # conf_index(matrix)
    matrixr1b=np.array(
        [[23,8,4],
         [ 9,8,0],
         [ 0,0,10]])

    matrixr2b=np.array(
        [[18,15,2],
         [ 5,12,0],
         [ 2,0,8]])

    conf_index(matrixr1a)
    out_put = "confuse_dispr"
    if not os.path.isdir(out_put):
        os.makedirs(out_put)
    save_path = os.path.join(out_put, "matrixr1a")
    confuse_plot(matrixr1a,save_path)

    save_path = os.path.join(out_put, "matrixr2a")
    confuse_plot(matrixr2a,save_path)

    save_path = os.path.join(out_put, "matrixr1b")
    confuse_plot(matrixr1b,save_path)

    save_path = os.path.join(out_put, "matrixr2b")
    confuse_plot(matrixr2b,save_path)


def ca2():
    matrixm1a = np.array(
        [[82,0,22],
         [19,0,12],
         [11,0,38]]
        )

    matrixm2a = np.array(
        [[85,2,17],
         [23,2,6],
         [11,2,36]])

    matrixm3a = np.array(
        [[85,2,17],
         [23,2,6],
        [11,2,36]])

    matrixm4a = np.array(
        [[90,9,5],
         [29,2,0],
        [15,2,32]])

    matrixm5a = np.array(
        [[82,22,0],
         [19,12,0],
        [6,0,43]])

    matrixm1b = np.array(
        [[4,19,12],
         [1,12,4],
        [0,3,7]])

    matrixm2b = np.array(
        [[15,1,19],
         [5,2,10],
        [1,0,9]])

    matrixm3b = np.array(
        [[33,2,0],
         [15,1,1],
        [3,0,7]])

    matrixm4b = np.array(
        [[31,0,4],
         [14,1,2],
        [1,0,9]],)

    matrixm5b = np.array(
        [[32,3,0],
         [13,3,1],
        [1,0,9]])

    conf_index(matrixm1a)
    print(matrixm1a)
    conf_index(matrixm1a)
    out_put = "confuse_dispm"
    if not os.path.isdir(out_put):
        os.makedirs(out_put)
    save_path = os.path.join(out_put, "matrixm1a")
    confuse_plot(matrixm1a,save_path)

    save_path = os.path.join(out_put, "matrixm2a")
    confuse_plot(matrixm2a,save_path)

    save_path = os.path.join(out_put, "matrixm3a")
    confuse_plot(matrixm3a,save_path)

    save_path = os.path.join(out_put, "matrixm4a")
    confuse_plot(matrixm4a,save_path)

    save_path = os.path.join(out_put, "matrixm5a")
    confuse_plot(matrixm5a,save_path)

    save_path = os.path.join(out_put, "matrixm1b")
    confuse_plot(matrixm1b, save_path)

    save_path = os.path.join(out_put, "matrixm2b")
    confuse_plot(matrixm2b, save_path)

    save_path = os.path.join(out_put, "matrixm3b")
    confuse_plot(matrixm3b, save_path)

    save_path = os.path.join(out_put, "matrixm4b")
    confuse_plot(matrixm4b, save_path)

    save_path = os.path.join(out_put, "matrixm5b")
    confuse_plot(matrixm5b, save_path)
if __name__ == '__main__':
    ca1()
    ca2()

