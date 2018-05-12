import matplotlib.pyplot as plt
import pdb
import argparse


def argParser():
	parser = argparse.ArgumentParser(description='PyTorch Plot Progress')
	parser.add_argument('--file_name', default='')
	return parser.parse_args()


def main():
	args = argParser()
	train_accuracy=[]
	test_accuracy=[]
	train_loss=[]
	with open(args.file_name) as f:
		for line in f:
			if 'Final Summary' in line:
				train_loss.append(float(line[:-1].split(' ')[-1]))
			elif 'Train Accuracy of the network' in line:
				train_accuracy.append(float(line[:-1].split(' ')[-2]))
			elif 'Test Accuracy of the network' in line:
				test_accuracy.append(float(line[:-1].split(' ')[-2]))

	
	plt.plot(train_accuracy)
	plt.plot(test_accuracy)
	plt.show()
	plt.plot(train_loss)
	plt.show()



if __name__ == '__main__':
	main()