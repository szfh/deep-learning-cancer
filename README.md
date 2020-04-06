# Breast cancer deep learning :snake:

### A project for the jHub Coding Scheme (JCS)

## Plan

1. Import dataset :heavy_check_mark:
1. **Data preprocessing** :heavy_check_mark:
	1. Create `X` and `y` :heavy_check_mark:
	1. Replace datetimes with strings :heavy_check_mark:
	1. Replace ranges with mid point :heavy_check_mark:
	1. Handle missing values :heavy_check_mark:
	1. Encode catergorical variables :heavy_check_mark:
		1. LabelEncoder :heavy_check_mark:
		1. OneHotEncoder :heavy_check_mark:
			1. Avoid the dummy variable trap :heavy_check_mark:
	1. Feature scaling :heavy_check_mark:
	1. Split to `training` and `test` set :heavy_check_mark:
1. **Compile the model** :heavy_check_mark:
	1. Import Keras :heavy_check_mark:
	1. Initialise the ANN :heavy_check_mark:
	1. Add the input layer :heavy_check_mark:
	1. Add the first hidden layer :heavy_check_mark:
	1. Add the second hidden layer :heavy_check_mark:
	1. Add the output layer :heavy_check_mark:
	1. Compile the ANN :heavy_check_mark:
	1. Fit the ANN to the training set :heavy_check_mark:
1. **Make predictions** :heavy_check_mark:
	1. Predict the test results :heavy_check_mark:
	1. Make the confusion matrix :heavy_check_mark:
1. Improve model :large_orange_diamond:
1. Comments and notes :large_orange_diamond:
1. Check and simplify code
1. Run on Colab

## Task

1. Using Python, create the best performing neural networks algorithm you can to predict recurrence rates of breast cancer based upon the variables provided in the attached breast cancer spreadsheet.
1. Document:
	1. the settings you tested (and rationale for the strategy you took) along the way to optimal performance.
	1. a screenshot of you using the trained algorithm to make a prediction on an unseen piece of data.

## Resources

1. https://towardsdatascience.com/data-preprocessing-in-python-b52b652e37d5
1. https://towardsdatascience.com/assessing-the-quality-of-data-e5e996a1681b
