import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats
import statsmodels.formula.api as smf
import pickle
import os
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

def load_dataset():
	return pd.read_csv('winequality-white.csv', delimiter=';')

def create_new_sample():
	df = load_dataset()
	num_samples = math.floor(len(df)*0.1)
	return df.sample(n=num_samples)

def save_sample(sample):
	with open("sample.p", "wb") as file:
		pickle.dump(sample, file)

def save_sample_to_csv(sample):
	sample.to_csv('sample.csv', index=False)

def prompt_load_sample():
	return input("Load old sample? ")

def load_sample(path=None):
	if path is None:
		path = "sample.p"
	if (os.path.exists(path)):
		with open(path, "rb") as file:
			return pickle.load(file)
	else:
		print('Sample not found')

def create_linear_regressors(sample, X, Y, x_axis, y_axis):
	linear_regressor = LinearRegression()
	linear_regressor.fit(X, Y)

	ols = smf.ols(formula=f'Q("{y_axis}") ~ Q("{x_axis}")', data=sample)
	ols_result = ols.fit()

	return linear_regressor, ols_result

def create_scatter_plot(X, Y, Y_pred):
	label = f'{x_axis.capitalize()} vs {y_axis}'
	scatter_fig = plt.figure(num=label)
	scatterplot = scatter_fig.add_subplot(111)
	sc = scatterplot.scatter(X, Y, c=Y, cmap=cm.gist_rainbow)
	plt.colorbar(sc)
	scatterplot.plot(X, Y_pred, c='Black', label="Least squares regression line")
	scatterplot.set_ylabel(y_axis.capitalize())
	scatterplot.set_xlabel(x_axis.capitalize())
	scatterplot.legend(loc="upper right")
	scatterplot.set_title(label=label)
	plt.savefig(f'{label}.png', transparent=True)

def create_residual_plot(X, Y, Y_pred):
	label = 'Residual plot'
	residual_fig = plt.figure(num=label)
	residual_plot = residual_fig.add_subplot(111)
	sc = residual_plot.scatter(X, Y-Y_pred, c=Y-Y_pred, cmap=cm.gist_rainbow)
	plt.colorbar(sc)
	residual_plot.axhline(y=0, c='Black')
	residual_plot.set_ylabel('Residual')
	residual_plot.set_xlabel(x_axis.capitalize())
	residual_plot.set_title(label=label)
	plt.savefig(f'{label}.png', transparent=True)

def create_residual_histogram(Y, Y_pred):
	label = 'Histogram of residuals'
	histogram_fig = plt.figure(num=label)
	histogram_plot = histogram_fig.add_subplot(111)
	histogram_plot.hist(Y-Y_pred)
	histogram_plot.set_ylabel('Frequency')
	histogram_plot.set_xlabel('Residual')
	histogram_plot.set_title(label=label)
	plt.savefig(f'{label}.png', transparent=True)

def create_residual_normal_dist(Y, Y_pred):
	label = 'Distribution of residuals'
	normal_dist_fig = plt.figure(num=label)
	normal_dist_plot = normal_dist_fig.add_subplot(111)
	normal_dist_plot.hist(Y-Y_pred, density=True, alpha=0.6, color='g')

	mu, std = norm.fit(Y-Y_pred)
	x = np.linspace(min(Y-Y_pred), max(Y-Y_pred), 100)
	p = norm.pdf(x, mu, std)

	normal_dist_plot.plot(x, p, 'k', linewidth=2)
	normal_dist_plot.set_xlabel('Residual')
	normal_dist_plot.set_title('Fit results: mu = %.2f,  std = %.2f' % (mu, std))
	plt.savefig(f'{label}.png', transparent=True)

def print_statistics(linear_regressor, ols_result, X, Y, num_samples):
	r2_value = linear_regressor.score(X, Y)
	r_value = math.sqrt(r2_value)
	slope = linear_regressor.coef_.flatten()[0]
	y_int = linear_regressor.intercept_[0]
	std_error = ols_result.bse[f'Q("{x_axis}")']

	degrees_of_freedom = num_samples - 2
	sig_level = 0.05
	t_value = abs(scipy.stats.t.ppf(q=sig_level, df=degrees_of_freedom))
	confidence_interval_lower = slope - t_value * std_error
	confidence_interval_upper = slope + t_value * std_error

	print("Conditions needed for inference:")
	print("Linear: The scatterplot shows a clear linear form. The residual plot shows random scatter.")
	print("Independent: Observations are independent from one another.")
	print("Normal: The histogram of the residuals is unimodal and mostly bell-shaped.")
	print("Equal SD: The residual plot shows similar amounts of scatter for each x.")
	print("Random: The observations were randomly sampled.")
	print("")

	print(f"Population size: {len(load_dataset())}")
	print(f"Sample size (n): {num_samples}")
	print(f"df: {degrees_of_freedom}")
	print(f"R-value: {r_value}")
	print(f"Slope: {slope}")
	print(f"Y-intercept: {y_int}")
	print(f"t-value: {t_value}")
	print(f"Standard error of slope: {std_error}")
	print(f"Confidence interval: ({confidence_interval_lower}, {confidence_interval_upper})")
	print("")

	print(f"Conclusion: We are {(1-sig_level)*100}% confident that the interval from {confidence_interval_lower} to {confidence_interval_upper} captures the slope of the true regression line relating the {y_axis} y and {x_axis} x of Portuguese Vinho Verde wine.")

if __name__ == "__main__":
	load_ans = prompt_load_sample()
	if load_ans == "y" or load_ans == "Y" or load_ans == "Yes" or load_ans == "yes":
		path_ans = input("Enter path or leave blank for default (sample.p) ")
		if len(path_ans) > 0:
			sample_df = load_sample(path_ans)
		else:
			sample_df = load_sample()
		save_sample_to_csv(sample_df)
	else:
		print("Creating new sample")
		sample_df = create_new_sample()
		save_sample(sample_df)
		save_sample_to_csv(sample_df)
	
	num_samples = len(sample_df)
	
	x_axis = 'free sulfur dioxide'
	y_axis = 'total sulfur dioxide'
	
	X = sample_df[x_axis].values.reshape(-1, 1)
	Y = sample_df[y_axis].values.reshape(-1, 1)

	linear_regressor, ols_result = create_linear_regressors(sample_df, X, Y, x_axis, y_axis)

	Y_pred = linear_regressor.predict(X)

	create_scatter_plot(X, Y, Y_pred)
	create_residual_plot(X, Y, Y_pred)
	create_residual_histogram(Y, Y_pred)
	create_residual_normal_dist(Y, Y_pred)
	print_statistics(linear_regressor, ols_result, X, Y, num_samples)

	plt.show()