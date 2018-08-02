# Fake_Reviews: Supervised Machine Learning Techniques 

This project has two components: the detection and generation of fake reviews 

Detection: The relevant features were extracted from the review (word count, frequency, capitalization, etc). The data was stored in a matrix, where each row in the matrix was a different review and each column was a different feature. Using the labeled data and a Support Vector Machine (SVM), another weighted matrix was created that was able to predict based on the feature matrix, whether the corresponding review was fake or real. 

Generation: We generated fake reviews by picking a random review in the data set as our "base review". The sentences in this review would be converted to a vector (representing the word count/frequency) using the Term Frequency - Inverse Document Frequency (tf-idf) library. The other sentences in the "sentence pool" ( the collection of sentences from the other reviews in the batch) were also converted to vectors. Each sentence in the base review was then replaced from one in the "sentence pool", based on how close the vectors were to each other (and the cosine function was used to determine similarity between the sentence vectors). 

