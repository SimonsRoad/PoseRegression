# conv1 filter visualization 
## towncenter
- filter size: 3x64x5x5 (in x out x width x height)
- visualize filters learned for 10 locations 

## Notes
- Some of SPNs were not trained very well. For example, at location 8, the weights are mostly zeros. Likewise, location 4 and 6 have many zero values. 
- Before seeing the learned filters, I actually expected that the filters at the bottom will show mostly low-level features like gabor filters. I wonder if it is 1) because our task is not just detection nor simply pose estimation, but the combination of detection, pose estimation, segmentation, 2) what other pose estimation baseline models will produce for the features. One thing that I would do for sure is to check and compare activation maps for some test images.  


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


