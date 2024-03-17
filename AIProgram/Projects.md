# Technion - AI Program Projects

[![](./FixelAlgorithmsLogo.png)](https://fixelalgorithms.gitlab.io/)
![](https://i.imgur.com/kvThExG.png)

[![Visitors](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FRoyiAvital%2FStackExchangeCodes&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visitors+%28Daily+%2F+Total%29&edge_flat=false)](https://github.com/FixelAlgorithmsTeam/FixelCourses)


## Project Options 

Contains list of projects for the choice of the students.  

<!-- https://stackoverflow.com/a/72327818 -->
> [!NOTE]
> This list is only a suggestion. The student may chose a different subject with coordination with the the guide.

### Classic Machine Learning

 -  Improve `Predictive Power Score (PPS)` implementation.
 -  Predict 1:1 Match Result by Past Data (Tennis, Chess, etc...).
 -  Recommendation system using Decision Trees.
 -  Predict the Probability of a Worker to Resign   
    See the [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset).  
    See [What Is Your Model Hiding? A Tutorial on Evaluating ML Models](https://www.evidentlyai.com/blog/tutorial-2-model-evaluation-hr-attrition).
 - 


**Remark**: The projects above can be solved with Deep Learning as well.

### Computer Vision

 - Tracking football players during play.
 - Identifying events in a video stream.
 - Measuring heart rate and respiration rate from a video.
 - Building a drone tracker.
 - Continuously verifying the person identity over a video.
 - Analyzing stress level via audio and video.
 - Sign Language Parsing.  
   See [Google - Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs).  
   [Live Coding by Rob Mulla](https://www.youtube.com/watch?v=DTQA8KIWWhY).
 - Image Denoising  
   Replicate the results of state of the art Denoising algorithm.
 - 

### Language Processing

 - Text Sentiment Labeling  
   For instance see [Sentiment Classifier for User Requests in the Banking Domain](https://rubrix.readthedocs.io/en/master/tutorials/01-labeling-finetuning.html).
 - 


### Data Resources

 - [CodaLab](https://codalab.lisn.upsaclay.fr) - ML competitions.
 - [CodaBench](https://www.codabench.org) - ML competitions (Refinement of `CodaLab`).
 - [Kaggle](https://www.kaggle.com) - ML competitions.
 - [Real World ML systems](https://www.evidentlyai.com/ml-system-design) - List of data sets accompanied by a project information.
 - [TidyTuesday](https://github.com/rfordatascience/tidytuesday) - A weekly social data project. Many datasets with links to EDA's. There are videos on YouTube (Search for `Tidy Tuesday`).

## Project Structure

The project should have the following structure:

 1. Problem Definition
   -  Define the problem you’re trying to solve.  
      Define it by the real world problem and the solution.  
      For example, in order to create an online site for users to estimate the price of their car we build a system which estimates the price of the car by its features.
   -  Define the scoring to be used and a reasonable target.  
      For instance, we can use the Mean Absolute Error as the scoring method. The objective MAE is 2000 [NIS].  
      Or we can use Accuracy as the scoring and the objective is 90%.
   -  Define the data you’d like to have. Compare it to the data you actually have. Both in quantity and features.
   -  Define the conditions to use the algorithm.  
      For example, in the above we’ll say it is the engine of a web site. It should be able to handle 1000 entries / hour.
 2. Data Exploration
   -  Present data. Explain why it was presented in the manner chosen.  
      For example, Use Histograms / Violin Plots to present data which is statistical and infer its Dynamic Range, Spread, Histogram of Distances, etc...
   -  Make some conclusion on the data based on the visual information.  
      For example, look at the corr() plot and say something about the linear relationship between different features.
   -  Try estimating the difficulty of achieving the target score.
 3. Data Pre Processing
   -  Handle missing data. Explain the method used (Removal, replacement, etc...).
   -  Remove outliers. Explain the model used to infer an outlier.
 4. Feature Engineering
   -  If data requires, build new features from existing features. Analyze their contribution.
   -  Do estimation about feature importance. Choose the features relevant to the task.
 5. Supervised Learning (Regression / Classification)
   -  Choose the methods which are applicable to your problem. Explain your choices.
   -  Build the testing environment. Explain the choices made (About the size of the test set, which type of cross validation, Grid search, etc...).
   -  Apply the chosen method. Compare the scoring. Try different scoring and explain results. Use cross validation to search for optimal hyper parameters.
   -  Try a method you excluded on (a). Compare it to the results. Was it a good decision to exclude it?
   -  Go back to step (2) and use the insight gained.
 6. Summary
   -  Present summary of results. Declare whether you failed or succeeded to reach the objective.
   -  Make some remarks about what it will take to move the project into production.  
      Address things like the scalability of the model, how would you handle new data, etc...
   -  Provide some recommendations for future work (Extensions).
 7. Presentation
   -  Should be ~10-15 slides.
   -  Aim for 30 Minutes.
   -  Rehearsal the presentation.

> [!TIP]
> Good project is one the doers learned from doing it, not by the score of it.