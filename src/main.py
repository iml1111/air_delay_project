import preprocess
import learning

def main():
	df = preprocess.proc('AFSNT.csv')
	learning.proc(df)

if __name__ == '__main__':
	main()
