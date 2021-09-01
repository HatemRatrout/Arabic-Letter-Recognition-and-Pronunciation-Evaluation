# Arabic Letter Recognition and Pronunciation Evaluation
### Abstract 
Arabic speech recognition that provides an automated framework to recognize Arabic alphabet using deep learning (convolutional neural network) algorithms that are able to learn hidden patterns from the data by themselves, combine them together, and build much more efficient decision rules. 

For a long time many students were facing the problem of pronouncing Arabic letters and it increased with e-learning. So from here our idea came to teach them how to pronounce letters correctly through a game in an enjoyable way.
Our educational game was designed using the Unity engine. It includes four levels and the most important of all the levels is the fourth level of the game which contains the main idea of our project and it is about Arabic speech recognition.

In this project we investigate the use of the deep convolutional neural network to recognize the Arabic letters. We used 4 different models. First we built our cnn layers, the second model is AlexNet, the third model is vgg16 uses transfer learning once in both ways with and without tuning and the last model is YamNet using transfer technique.

To evaluate the performance of the system, we compared the results of these models for Arabic letters. In the baseline method, the experimental results show for 28 letters that our cnn layers and YamNet achieve an accuracy of 74% and 83.47% respectively.

And for just 7 letters AlexNet, vgg16 without tuning, vgg16 with tuning achieve an accuracy of 55%, 77.94%, 86.0% respectively. The performance analysis shows that YamNet outperforms for 28 letters and achieves an accuracy of 83.47% and vgg16 with tuning outperforms for just 7 letters and achieves an accuracy of 86.0%.

## Technologies used:
- Python
- Keras framework
- Tensorflow
- Matlab
  - Deep Learning Toolbox
  - Audio Toolbox
- C#
- Unity Engine

## The steps we followed to build the recognition system:
1. Collected the data set for the 28 Arabic letters.
2. Pre-processed the audio dataset.
3. Split the audio dataset to train, test and validation sets.
4. Extracted features from the audio.
5. We had 2 form datasets one is MFCCS.
6. The second one is mel-spectrogram. 
7. Then we used these data as inputs for the models we use and built.
8. Then trained the model with different epochs and batch size and learning rate.

## Results:
The result of training differs from model to another, they are vary between good and bad results.
- ### CNN
<img width="500px" src="https://user-images.githubusercontent.com/76398557/131649019-69e52c4a-2bbc-4427-bd80-16fec862b731.png" >  
<img width="500px" src="https://user-images.githubusercontent.com/76398557/131649091-398ab5a3-e453-4b23-a238-c14772c27042.png" >

- ### Alexnet
<img width="500px" src="https://user-images.githubusercontent.com/76398557/131649520-1c45950d-583a-46a0-b62f-96d3279c80fa.png" >  
<img width="500px" src="https://user-images.githubusercontent.com/76398557/131649527-8fb3296c-91f8-4ae7-9448-fa9476bd27e7.png" >  

- ### Yamnet
<img width="700px" src="https://user-images.githubusercontent.com/76398557/131649737-f725980a-cc5d-4bd1-a216-bc4c80e492c0.png" >  

## Interface
The best way to teach children is the **'trial and error'** method. When the child is placed in an interactive environment in which he can try to learn right and wrong, this is much better than the method of **'indoctrination'** or any other methods, and this is what we have adopted in our project.  
In order to achieve this goal, we have made an interactive game for children through the Unity game engine, consisting of 4 levels, and in various cartoon shapes and colors, close and loved by children.  
We have also diversified these forms and stages so that the child does not feel bored.

- ### Main Menu
<img width="700px" src="https://user-images.githubusercontent.com/76398557/131697187-facfaa01-c329-4332-9f07-32a8f8b43c55.png" >  

- ### Level 1 (letters and words – الحروف والكلمات)
The main goal of the first level is for the child to listen to the correct pronunciation of the Arabic letter, and then present this letter with an example (and the intended letter in the example is shaded in a different color from the rest of the word). 
These examples have been diversified (animal, inanimate, and plant). 
&nbsp; <br>  
<img width="700px" src="https://user-images.githubusercontent.com/76398557/131697933-43575d91-3fb3-402b-90af-cd8195566a75.png">  
<img width="700px" src="https://user-images.githubusercontent.com/76398557/131697946-797f4efb-2775-4b66-b0b8-87564fb435c7.png" >  

- ### Level 2 (Letters Pronunciation – نطق الحروف)
The second level is the most important level in the game, where it is checked whether the child pronounces the letter correctly or not, here our model is called and the processing process is done and results are shown. 
&nbsp; <br>  
<img width="700px" src="https://user-images.githubusercontent.com/76398557/131698087-4874d46e-def3-4c72-99cb-959131e13242.png" >  

- ### Level 3 (Missing Letter – الحرف الناقص)
At this level, there are words that are missing the first letter of them, the child must put the appropriate letter in the appropriate word, so the goal of this stage is for the child to link the letters that he learned with words or examples from the Arabic language.
&nbsp; <br>  
<img width="700px" src="https://user-images.githubusercontent.com/76398557/131699277-a530a124-2c32-4f04-8a86-946fa0133e8b.png" >  
<img width="700px" src="https://user-images.githubusercontent.com/76398557/131699283-fc76a5f4-efa8-4f3e-b827-2297de5490b9.png" >  

- ### Level 4 (Arabic Alphabet Board – كتابة الحروف)
After the child has learned the correct pronunciation of the Arabic letters, and their uses in some words, it is time to learn the correct writing of these letters.
&nbsp; <br>  
<img width="700px" src="https://user-images.githubusercontent.com/76398557/131699790-c6610c4c-416b-47ad-85b8-8427a2e04097.png" >  
<img width="700px" src="https://user-images.githubusercontent.com/76398557/131699797-665ecd40-b879-4e50-827a-b8426b8b4244.png" >  
<img width="700px" src="https://user-images.githubusercontent.com/76398557/131699803-df422a91-29a9-4f5a-9f28-e76ca79f24d6.png" >  

## Team members:
- May Hannon (MayHannon)
- Hatem Ratrout (HatemRatrout)
- Dalal Yassin
