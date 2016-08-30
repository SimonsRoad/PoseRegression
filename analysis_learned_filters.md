# conv1 filter visualization 

## Settings
- dataset 1: towncenter
- filter size: 3x64x5x5 (in x out x width x height)
- visualize filters learned for 10 locations 

## Notes
- Some of SPNs were not trained very well. For example, at location 8, the weights are mostly zeros. Likewise, location 4 and 6 have many zero values. 
- Learned filters do not look similar to each other. This means that SPNs learned something different (assuming the training is equally successful). Note that the difference between SPNs are scale and background: A comparison between actual scales and background images should be made.
- Learned filters do not look like gabor filters, which are usually seen from conv1s of well-trained models for some image classification tasks. I wonder if our SPNs are also supposed to look like those. In other words, is it possible that our setup of multi-task (detection, pose esetimation, segmentation) may have affected and learned differently? Or should I suspect that learning is simply not done well?
- I wonder what other well-trained pose estimation networks produce for conv1 filters. 
- The next thing I will do is check and compare activation maps for layers at multiple stages given some test images, instead of filters'' weights. 



## Visualizations
<table>
    <tr> 
        <td> location 1 </td>
        <td> location 2 </td>
        <td> location 3 </td>
    </tr>
    <tr>
        <td> <img src="visualize_filters/conv1_results/conv1_weights_loc1.png" height="280"> </td>
        <td> <img src="visualize_filters/conv1_results/conv1_weights_loc2.png" height="280"> </td>
        <td> <img src="visualize_filters/conv1_results/conv1_weights_loc3.png" height="280"> </td>
    </tr>
    <tr> 
        <td> location 4 </td>
        <td> location 5 </td>
        <td> location 6 </td>
    </tr>
    <tr>
        <td> <img src="visualize_filters/conv1_results/conv1_weights_loc4.png" height="280"> </td>
        <td> <img src="visualize_filters/conv1_results/conv1_weights_loc5.png" height="280"> </td>
        <td> <img src="visualize_filters/conv1_results/conv1_weights_loc6.png" height="280"> </td>
    </tr>
    <tr> 
        <td> location 7 </td>
        <td> location 8 </td>
        <td> location 9 </td>
    </tr>
    <tr>
        <td> <img src="visualize_filters/conv1_results/conv1_weights_loc7.png" height="280"> </td>
        <td> <img src="visualize_filters/conv1_results/conv1_weights_loc8.png" height="280"> </td>
        <td> <img src="visualize_filters/conv1_results/conv1_weights_loc9.png" height="280">
    </tr>
    <tr> 
        <td> location 10 </td>
    </tr>
    <tr>
        <td> <img src="visualize_filters/conv1_results/conv1_weights_loc10.png" height="280"> </td>
    </tr>
</table>


