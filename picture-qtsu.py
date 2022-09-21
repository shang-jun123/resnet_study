import numpy as np
import cv2
import matplotlib.pyplot as plt
for i in range(600):
    path="D:\\git\\resnet-study\\datasets\\origin_pvdata6\\scratchs"
    path2 = "D:\\git\\resnet-study\\datasets\\origin_pvdata6\\scratchs2"
    path1=path+"\\"+str(i)+".jpg"
    path3 = path2 + "\\" + str(i) + ".jpg"
    img = cv2.imread(path1,cv2.IMREAD_GRAYSCALE)
    img_mat = img
    img_mat = img_mat.astype(np.uint8)
    threshold,img_mat = cv2.threshold(img_mat,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print(threshold)
    # plt.imshow(img_mat, cmap='gray')
    # plt.show()
    cv2.imwrite(path3,img_mat)

# ######我们先制造一个200x200的图片用于二值化实验#######
# def get_test_img():
#     img_mat = np.zeros((200,200),dtype=np.uint8)# 记得设置成整数，不然opencv会将大于1的浮点数全显示成白色
#     for row in range(200):
#         for col in range(200):
#             img_mat[row][col] = col
#     return img_mat
# img = cv2.imread(r'D:\git\resnet-study\spv_resnet\pictures\dark.jpg',cv2.IMREAD_GRAYSCALE)
# img_mat = img
# plt.imshow(img_mat,cmap='gray')# 显示图片
# plt.xlabel("raw img")
# plt.show()
# # 调用cv2中的otsu库
# #img_mat = get_test_img() # 这是实验1中的那个函数
# img_mat = img_mat.astype(np.uint8)
# threshold,img_mat = cv2.threshold(img_mat,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# print(threshold)
# plt.imshow(img_mat,cmap='gray')
# cv2.imwrite('messigray.png',img)
# plt.show()
