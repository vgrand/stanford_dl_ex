import re

"""
Formatting the housing.data file to replac all spaces by commas where appropriate:
 - Get rid of all spaces starting lines
 - Replace the remaining spaces with commas
"""

with open('housing.data', 'rb') as f:
	with open('housing.csv', 'wb') as g:
		g.write( re.sub( ' +', ',' ,re.sub( '^ +', '', re.sub('\n +', '\n', f.read()))))