import os
from PIL import Image
import random
import numpy as np
random.seed(42)
np.random.seed(42)
from PIL import ImageEnhance
from skimage.feature import hog,local_binary_pattern

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import dump



def load_and_organise_data():
  bad_image=[]
  glass_image=[]
  metal_image=[]
  paper_image=[]
  plastic_image=[]
  path_to_trash="C:\\Users\\Omojire\\Desktop\\recycling_project.py\\trash"
  for filename  in sorted(os.listdir(path_to_trash)):
     label_path=os.path.join(path_to_trash,filename)
     for Imagefile in sorted(os.listdir(label_path)):
         image_path=os.path.join(label_path,Imagefile)
         try:
             Image.open(image_path)
             if filename=="glass":
                 glass_image.append(image_path)
                
             elif filename=="metal":
                 metal_image.append(image_path)
                 
             elif filename=="paper":
                 paper_image.append(image_path)
                 
             elif filename =="plastic":
                 plastic_image.append(image_path)
              
         except :
             bad_image.append(image_path)
             print(bad_image)
             continue
         
  random.shuffle(glass_image)
  glass_image=glass_image[:410]
  
  random.shuffle(metal_image)
  metal_image=metal_image[:410]
   
  random.shuffle(plastic_image)
  plastic_image=plastic_image[:410]
   
  random.shuffle(paper_image)
  paper_image=paper_image[:410]
  
  return glass_image,paper_image,plastic_image,metal_image

glass_image,paper_image,plastic_image,metal_image=load_and_organise_data()

def resize_image(image_path):
    '''convert image to grayscale and resize'''
    resized_images = []
    for path in image_path:
        image = Image.open(path)
        image = image.convert('L')
        image = image.resize((64,64))
        resized_images.append(image)
    return resized_images              
              
def normalization():
    ''' convert grayscale images to pixels using numpy and then normalise grayscale pixels from
    0(white)-255(black) to 0(white)-1(black) by dividing by 255 to speed up our algorithm
    by ensuring faster convergence '''
    

    resized_glass=resize_image(glass_image)
    resized_paper=resize_image(paper_image)
    resized_plastic=resize_image(plastic_image)
    resized_metal=resize_image(metal_image)
    
    # Split resized images 80-20 BEFORE augmentation
    glass_train, glass_test = train_test_split(resized_glass, test_size=0.20, random_state=42)
    paper_train, paper_test = train_test_split(resized_paper, test_size=0.20, random_state=42)
    plastic_train, plastic_test = train_test_split(resized_plastic, test_size=0.20, random_state=42)
    metal_train, metal_test = train_test_split(resized_metal, test_size=0.20, random_state=42)
              

    pixel_images=[]
    
    glass_pixel=[np.array(img)/255.0 for img in  glass_train]
    paper_pixel=[np.array(img)/255.0 for img in paper_train]
    plastic_pixel=[np.array(img)/255.0 for img in plastic_train]
    metal_pixel=[np.array(img) /255.0 for img in metal_train]
    
    
    pixel_images.append(glass_pixel)
    pixel_images.append(paper_pixel)
    pixel_images.append(plastic_pixel)
    pixel_images.append(metal_pixel)
    
    # Keep test images separate for later
    test_pixel_images = []
    glass_test_pixel = [np.array(img)/255.0 for img in glass_test]
    paper_test_pixel = [np.array(img)/255.0 for img in paper_test]
    plastic_test_pixel = [np.array(img)/255.0 for img in plastic_test]
    metal_test_pixel = [np.array(img)/255.0 for img in metal_test]
    
    test_pixel_images.append(glass_test_pixel)
    test_pixel_images.append(paper_test_pixel)
    test_pixel_images.append(plastic_test_pixel)
    test_pixel_images.append(metal_test_pixel)
    
    ''' pixel_images(resized_images converted into pixels),then resized_images(just grayscale not pixels
    yet,we need it for augmentation-augmentation works on raw images not pixels)',
    test_pixel_images(we separate this from augumenatation to prevent data leakage,so we get a
    true accuracy ,this helps the model to be more accurate on real world data )'''

    

    return pixel_images, [glass_train, paper_train, plastic_train, metal_train], test_pixel_images

pixel_images, resized_images, test_pixel_images = normalization()
 

        
def image_augmentation():
    class_images=['glass','paper','plastic','metal']
    augmented_images=[]
    all_images=[]
    all_labels=[]
    
    '''if class_index == 0 then class is glass etc.'''
    
    
    for class_index,class_list in enumerate(pixel_images):
       for img in class_list: 
        all_images.append(img)
        all_labels.append(class_index)
    
    
    for class_index,image_list in enumerate(resized_images):
      for image in image_list:
          
        rotated= image.rotate(15)
        augmented_images.append(np.array(rotated)/255.0)
        all_labels.append(class_index)
    
        horizontal_flip=image.transpose(Image.FLIP_LEFT_RIGHT)
        augmented_images.append(np.array(horizontal_flip)/255.0)    
        all_labels.append(class_index)
        
        vertical_flip=image.transpose(Image.FLIP_TOP_BOTTOM)
        augmented_images.append(np.array(vertical_flip)/255.0)
        all_labels.append(class_index)
        
        width,height=image.size
        crop=image.crop((10,10,width-10,height-10))
        crop=crop.resize((64,64))
        augmented_images.append(np.array(crop)/255.0)
        all_labels.append(class_index)
        
        brightness=ImageEnhance.Brightness(image).enhance(1.5)
        augmented_images.append(np.array(brightness)/255.0)
        all_labels.append(class_index)
        
        contrast=ImageEnhance.Contrast(image).enhance(1.5)
        augmented_images.append(np.array(contrast)/255.0)
        all_labels.append(class_index)
      
     
    all_images.extend(augmented_images)
       
    return all_images,all_labels


def feature_engineering():#capture shape and patterns
    all_images,all_labels=image_augmentation()
    all_features=[] #hog vectors + local binary patterns(LBP)
    
    '''We will use HOG for feature extraction,mainly to capture shapes and edges of the trash materials
    HOG works best on grayscale image where colour isn't affecting it(we only care about intensity) 
    We will use something called sobel operators to capture the edges and get the gradients. 
    '''
    
    '''Angle  tells us the direction of the edge and it  has been converted to degrees in
    the range of 0 to 180'''
    
    
    '''We will divide the image into small cells i.e 8x8 and build histogram of
    gradient angles for each cell in n_bins where each cell has a bin_size'''
    
    for idx,each_image in enumerate(all_images):        
    
      if len(each_image.shape)==3:
            each_image=np.mean(each_image,axis=2)
        
      image_float=np.array(each_image,dtype=np.float32)#for hog
      image_int=(each_image*255).astype(np.uint8)
      '''
      SOBEL PARAMETERS
      xG=cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3) #horizontal-gradient
      yG=cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3) #vertical-gradient
      magnitude=np.sqrt(xG**2+yG**2)  
      angle=np.arctan2(yG,xG)*(180/np.pi)%180 '''
        
      #optimized HOG
      hog_features = hog(image_float, orientations=12, pixels_per_cell=(16,16), 
                      cells_per_block=(2,2), feature_vector=True,transform_sqrt=True,
                      block_norm='L2-Hys')
      
      # Multi-radius LBP (uniform) concatenated
      lbp_parts = []
      for radius in (1, 2, 3):
          n_points = 8*radius
          lbp = local_binary_pattern(image_int, n_points, radius, method='uniform')
          lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2,
                                      range=(0, n_points + 2), density=True)
          lbp_parts.append(lbp_hist)
      lbp_feat = np.concatenate(lbp_parts)

      # GLCM (Haralick) texture props on quantized image
      img_q = (image_int // 8).astype(np.uint8)  # 32 levels
      glcm = graycomatrix(img_q, [1, 2], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                          levels=32, symmetric=True, normed=True)
      glcm_props = []
      for p in ('contrast','homogeneity','energy','correlation'):
          vals = graycoprops(glcm, p).ravel()
          glcm_props.extend(vals)
      glcm_feat = np.array(glcm_props, dtype=np.float32)

      combined_features=np.concatenate([hog_features, lbp_feat, glcm_feat])
      all_features.append(combined_features)
    return np.array(all_features),np.array(all_labels)

def extract_test_features():
    """Extract features from test set (no augmentation)"""
    test_features = []
    test_labels = []
    
    for class_index, class_list in enumerate(test_pixel_images):
        for img in class_list:
            test_labels.append(class_index)
            
            if len(img.shape) == 3:
                img = np.mean(img, axis=2)
            
            image_float = np.array(img, dtype=np.float32)
            image_int = (img * 255).astype(np.uint8)
            
            hog_features = hog(image_float, orientations=12, pixels_per_cell=(16,16), 
                              cells_per_block=(2,2), feature_vector=True, transform_sqrt=True,
                              block_norm='L2-Hys')

            # Multi-radius LBP
            lbp_parts = []
            for radius in (1, 2, 3):
                n_points = 8 * radius
                lbp = local_binary_pattern(image_int, n_points, radius, method='uniform')
                lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2,
                                           range=(0, n_points + 2), density=True)
                lbp_parts.append(lbp_hist)
            lbp_feat = np.concatenate(lbp_parts)

            # GLCM features
            img_q = (image_int // 8).astype(np.uint8)
            glcm = graycomatrix(img_q, [1, 2], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                                levels=32, symmetric=True, normed=True)
            glcm_props = []
            for p in ('contrast','homogeneity','energy','correlation'):
                vals = graycoprops(glcm, p).ravel()
                glcm_props.extend(vals)
            glcm_feat = np.array(glcm_props, dtype=np.float32)

            combined_features = np.concatenate([hog_features, lbp_feat, glcm_feat])
            test_features.append(combined_features)
    
    return np.array(test_features), np.array(test_labels)

def train_model():
     # Get training features (with augmentation)
     X_train, y_train = feature_engineering() 
     
     # Get test features (without augmentation)
     X_test, y_test = extract_test_features()
     
     print(f"Training set size: {len(X_train)}")
     print(f"Test set size: {len(X_test)}")

     # Train model (best params from search)
     model = make_pipeline(
         StandardScaler(),
         PCA(n_components=None, random_state=42),
         SVC(kernel='rbf',
             C=16,
             gamma='scale',
             class_weight='balanced',
             cache_size=1000
         ))
        
     model.fit(X_train, y_train)
     score = model.score(X_test, y_test)
     
     print(f"Test accuracy: {score:.4f}")
     return model, score


def save_model(model,path:str='Recycling_project.joblib'):
    dump(model,path)
    print(f"model saved to {path}")
    

if __name__=='__main__':
    model,score=train_model()
    save_model(model)