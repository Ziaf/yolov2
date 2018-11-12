#imports and configuration
from detector import *
import models

%reload_ext autoreload
%autoreload 2

torch.set_printoptions(linewidth=200)
np.set_printoptions(linewidth=200)






device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
input_size = 608






model, cats = models.yolov2_coco(device, input_size, input_size)






test_single_image(model, cats, 'images/dog-cycle-car.png', input_size, device)



run_on_camera(model, input_size, cats, device)
