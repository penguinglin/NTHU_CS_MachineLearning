# NTHU_CS_MachineLearning - Handwritten Mathematical Expression Recognition

### Dependencies



| Module          | Version |
| --------------- | ------- |
| Tensor-flow GPU | 2.10    |
| Keras           | 2.10    |
| Numpy           | 1.26.4  |
| imageio         | 2.36.1  |
| opencv-python   | 4.10    |
| matplotlib      | 3.4.2   |
| tqdm            | 4.67    |
| tkinter         | 8.6     |

Operated with Python = 3.9

### Fetch checkpoints (Necessary before execution)

Download the two required file on this [link]([https://drive.google.com/drive/folders/13kkLBnhuWrnod1gcaI7KAo86qEKMRGVI?usp=drive_link](https://drive.google.com/drive/folders/1UTnCO9EIFA4HHc-CjIZ58-of4TQ3u7Qo?usp=sharing))

And copy the files into `/checkpoints/train`, then you could start using the application.

### Instructions

Direct execute `./model.py` then wait for the program to initialize and load checkpoint into the model.

Then the GUI should pop up.

![image](https://i.imgur.com/IzOBCJy.png)

Save: Save the current canva to "OutputFile.jpg".
Clean: Erase all the stroke on canva.
Pen: Pen mode, could paint on canva.
Eraser: Eraser mode, now cursor surves as a eraser that could eliminate the stroke.

In Tool Control section, you could adjust the width of your stroke and eraser.
In Draw Tool section, you're able to change your selected tool.

Also,we've provide different color scheme of the app and different window size if you wish.



> The details of this research can be found in [Report](ML_Report.pdf)

> Team members: Pei-Lin, Hua-Wei, I-Ping, Wei-Sheng, CTing-Jyun.
