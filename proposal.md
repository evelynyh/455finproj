# Team Imagenary: Where’s Waldo Image Detection #

Members: Kion Bidari, BT Lohitnavy, Davinder Singh, Evelyn Heckman

## Goal ##

Using images of isolated Waldos to learn his features and images of Waldo in a crowd/scene for validation and testing of the identification algorithm we would like to develop, we hope to highlight/outline Waldo, given any scene.

## Hypothesis ##

Using learned computer vision and machine learning concepts, we aim to develop an algorithm that accomplishes our goal by collecting a comprehensive dataset of Waldos and Waldo in crowded scenes, design an architecture for Waldo image recognition, and train the model’s parameters in order to fine-tune detection accuracy. The model should account for scene complexity and the specific Waldo features in order to maximize its utility.

## Methodology ##

We use machine learning to identify Waldo’s key features (face, shirt, hat, cane, etc.), learn Waldo’s face using eigenfaces, and with the learned features, search for likely matches of Waldo in an image.

## Breakdown of Milestones ##

### Waldo Classification ###

* Selection of data for classifying Waldo/not Waldo

* Model selection and hyperparameter tuning (Waldo’s key features learned here)

* Testing of Waldo classification model

### Search for Possible Waldos in an Image ###

* Implementation of a simple search algorithm

* Generate results from Waldo in crowded images (false positives, true positives, false negatives, true negatives, accuracy, and common blind spots within the algorithm)

### Optimize and Improve Waldo Search ###

* Selection of areas of interest (exclusion of clear non-Waldo entities)

* From these selections, parse and find the possible Waldos

* Generate another set of results and compare with simple search algorithm

* Compare our results with others’ algorithms
