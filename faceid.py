#import dependincies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

import cv2
import tensorflow as tf
from layer import L1dist
import os
import numpy as np

class CamApp(App):
    def build(self):
        self.img1=Image(size_hint=(1,.7))
        self.button=Button(text="verify", on_press=self.verify , size_hint=(1,.1))
        self.verification_label=Label(text="verification unintiated", size_hint=(1,.1))

        layout=BoxLayout(orientation='vertical')
        layout.add_widget(self.img1)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)
        
        self.capture=cv2.VideoCapture(0)
        
        Clock.schedule_interval(self.update,1.0/30.0)
        #import model
        self.model=tf.keras.models.load_model('facerecg.h5',custom_objects={'L1dist':L1dist})

        return layout
    def update(self,*args):
        ret,frame=self.capture.read()
        frame=frame[200:200+300,200:200+250, :]
            
        buf=cv2.flip(frame,0).tostring()
        img_texture=Texture.create(size=(frame.shape[1],frame.shape[0]),colorfmt='bgr')
        img_texture.blit_buffer(buf,colorfmt='bgr',bufferfmt='ubyte')
        self.img1.texture=img_texture
    def preprocess(self,file_path):
        b_img = tf.io.read_file(file_path)  # Read image file
        img = tf.io.decode_jpeg(b_img)      # Decode jpeg image
        img = tf.image.resize(img, (100, 100))  # Resize the image to 100x100
        img = img / 255.0                   # Normalize to [0, 1]
        return img
    
    def verify(self,*args):
        verification_limit=0.5
        detection_limit=0.5
        save_path=os.path.join('application_data','verification_images')
        ret,frame=self.capture.read()
        frame=frame[200:200+250,200:200+250, :]
        cv2.imwrite(save_path,frame)
        results=[]
        for image in os.listdir(os.path.join('application_data','verification_images')):
            input_img=self.preprocess(os.path.join('application_data','input_images','input_image.jpg'))
            validation_img=self.preprocess(os.path.join('application_data','verification_images',image))
            
            #result=self.model.predict(list(np.expand_dims([input_img,validation_img],axis=1)))
            result = self.model.predict([np.expand_dims(input_img, axis=0), np.expand_dims(validation_img, axis=0)])

            results.append(result)
            
        detection=np.sum(np.array(results)>detection_limit)
        verification=detection/len(os.listdir(os.path.join('application_data','verification_images')))
        verified=verification>verification_limit
        
        self.verification_label.text='verified' if verified== True else 'unverified'

        
if __name__=='__main__':
    CamApp().run()