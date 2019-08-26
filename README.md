
# 这是一个目标检测和目标分割增强的小工具，需要您事先标记一些图片，然后变化增强图片(支持LabelIMg和LabelMe标注的文件)，希望对您有所帮助！<br>
**包括3个python文件（rename_file.py、DataAugmentforLabelImg.py和DataAugmentforLabelMe.py）**<br>
1. rename_file.py能实现文件的重命名,注意修改文件的路径
2. DataAugmentforLabelImg.py能实现LabelImg标注后的图片的增强（包括模糊，亮度，裁剪，旋转，平移，镜像等变化）
3. DataAugmentforLabelMe.py能实现LabelMe标注后的图片的增强（包括模糊、亮度、平移、镜像等变化）

**注意：一些包的安装是必要的，比如Opencv_python等**
<br>
##*将您需要增强的图片放在对应的文件夹即可，具体可参考demo给出的图片和xml文件存放路径，您放入即可*
<br>
**保存结果：**<br>
1. 目标检测增强后的图片默认会保存在./data/Images2中，xml文件会保存在./data/Annotations2中（包括新的图片和xml文件）<br>
2. 目标分割增强后的图片默认会保存在./data2中（包括新的图片和json文件）
<br>

**实现这个工具参考了：**<br>
https://github.com/maozezhong/CV_ToolBox/blob/master/DataAugForObjectDetection/


##Demo
<br>

**LabelImg标定用于检测的图片,使用DataAugmentforLabelImg**


![image](https://github.com/pureyangcry/tools/blob/master/Imgs/d_demo.jpg)


增强结果：


![image](https://github.com/pureyangcry/tools/blob/master/Imgs/d_results.jpg)






**LabelMe标定用于检测的图片,使用DataAugmentforLabelMe**


![image](https://github.com/pureyangcry/tools/blob/master/Imgs/s_demo.jpg)


增强结果：


![image](https://github.com/pureyangcry/tools/blob/master/Imgs/s_results.jpg)