# Make a hashtag from the movie
# Download the tweets with the hashtag via Twitter API

from twython import Twython

APP_KEY = 'kkozCd1K5U94HyIUDiKHreyT1'
APP_SECRET = 'vTY0Tr8dMgJYh2IpvI2mNfqqFRT74QOZaN7LXuNyXtEC1Nn6PO'

twitter = Twython(APP_KEY, APP_SECRET, oauth_version=2)
ACCESS_TOKEN = twitter.obtain_access_token()

twitter = Twython(APP_KEY, access_token=ACCESS_TOKEN)

def downloadTweet(hashtag):

    search = twitter.search(q = '#GOT')
    tweets = search['statuses']
    #print(tweets)
    for tweet in tweets:
        tweet_id, tweet_text = tweet['id_str'], tweet['text']
        print(tweet_id, '\n', tweet_text)


def main():
    
	movies = open('imdb-data.csv', 'r')

	for line in movies:
		linelist = line.rstrip().split(",")
		moviename = linelist[1]
		splittedMoviename = moviename.split()
		hashtag = ["".join(splittedMoviename)]

		# If the moviename contains more than 2 words, we have to make an extra hashtag
		if len(splittedMoviename) > 2:
			hashtag.append("".join([x[0] for x in splittedMoviename]))

		print(hashtag)


if __name__ == "__main__":
    main()
