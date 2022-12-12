import numpy as np
import torch
import cv2, os
import pafy
from time import time
import matplotlib.pyplot as plt
print("Using torch", torch.__version__) # colab= 1.10.0+cu111



class Density_Estimation:
    '''
    Density estimation from videos in url
    Code credit:  https://github.com/akash-agni/Real-Time-Object-Detection/blob/main/Object_Detection_Youtube.py
    '''
    def __init__(self, url, out_file='output.avi'):
        ''' initialize the class'''
        self._URL     = url
        self.model    = self.load_model()
        self.out_file = out_file
        self.device   = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_video_from_url(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
        play = pafy.new(self._URL).streams[-1]
        assert play is not None
        return cv2.VideoCapture(play.url)

    def load_model(self):
        """
        Load pretrained model
        """
        PATH = '/home/muhammada/PDRA/works/cc_all/pretrained/ShanghaiTechPartB_CSRNet.pth'
        print(f' \n checkpoint exists? ==>  {os.path.isfile(PATH)}  \n')
        device   = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = torch.load(PATH, map_location=device) # map_location=self.device
        model.eval().to(device)
        print('Model loaded..')
        return model


    def create_density_map(self, frame):
        ''' create density map'''
        device      = 'cuda' if torch.cuda.is_available() else 'cpu'
        # frame is array of uint8
        _input      = torch.Tensor(frame).permute(2,0,1).unsqueeze(0).to(device) 
        density_map = self.model(_input/255) # return float32        
        count       = int(density_map.sum())
        print('Crowd count : ', count)
        
        ## Correct shapes (frame=(h,w,c), density_map=(h,w)
        density_map =  density_map.squeeze().detach().cpu().numpy()
        
        
        ## Scale up density map or scale down frame to match shapes
        # frame       = cv2.resize(frame, (density_map.shape[1], density_map.shape[0]))
        density_map = cv2.resize(density_map, (frame.shape[1], frame.shape[0]))
        
        ## Scale to 0-255
        # frame  = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        density_map = cv2.normalize(density_map,  None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        
        ## Overlay
        density_map = cv2.applyColorMap(density_map, cv2.COLORMAP_HOT) # Create a heatmap to make channels=3           
        output      = cv2.addWeighted(frame, 1, density_map, 1, 0)
        
        ## Stats
        output = cv2.putText(output, f'Total Count: {count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
        return frame, density_map, output


    def __call__(self):
        '''
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        return: void
        '''
        player = self.get_video_from_url()
        assert player.isOpened()
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        four_cc = cv2.VideoWriter_fourcc(*"MJPG") ## Four char video codec (See fourcc.org)
        out = cv2.VideoWriter(self.out_file, four_cc, 10, (x_shape, y_shape))  # cv2.VideoWriter(filename, fourcc, fps, frameSize, isColor = true)
        while True:
            start_time = time()
            ret, frame = player.read()  # frame is 'uint8, 0-255
            assert ret
            frame, density_map, output = self.create_density_map(frame)
            end_time    = time()
            
            #### Write frame
            cv2.imwrite('frame.jpg', frame)
            cv2.imwrite('density_map.jpg', density_map)
            cv2.imwrite('output.jpg', output)
            
            fps = 1/np.round(end_time - start_time, 3)
            # print(f"Frames Per Second : {fps}")
            out.write(output)
            
            cv2.imshow('output', output)
            cv2.waitKey(3000)
            
            


# url = 'https://www.youtube.com/watch?v=72NP6-KP7ek'
# url = 'https://www.youtube.com/watch?v=ng8Wivt52K0'
# url = 'https://www.youtube.com/watch?v=bDBMGxXS23c'
url = 'https://www.youtube.com/watch?v=-2FZhxoVwSw'
main = Density_Estimation(url, out_file='output.avi')
main()
