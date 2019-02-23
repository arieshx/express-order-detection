# express-order-detection
This project is mainly based ctpn(Detecting Text in Natural Image with Connectionist Text Proposal Network).I made adjustments and optimizations based on actual tasks.Since the implementation of this project has been a long time, many details have been forgotten, please forgive me.


## keyword of ctpn
* Method to realize mainly based on tensorflow
* Achieved　side-refinement
* Explored differenter loss Func，best choice:focal loss
## Main process of the project
1. Get a lot of tagged data
2. Training model：ctpn model and EAST model
3. Observe the results, summarize the merge rules, and merge the results of the two models
4. Iterative training model
5. Observe and analyze bad cases and correct it
## Main problem observed
* Positive and negative sample classification error
* The detection frame boundary is not accurate enough
* ...
## Main contribution
* Removed redundant detection frame
* Corrected the problem that the frame is too wide and too long
* Make up for the defect that the tensorflow version does not have this feature: side-refinement
## Results of the work
* Improved detection accuracy：For the new data in the daily business flow, we do not calculate the accuracy of it very accurately.
* But the profit margin of the entire business has increased by about 20 percent.

## Contact information
* arieshx.v@gmail.com
