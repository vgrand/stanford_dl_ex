import re

with open('housing.data', 'rb') as f:
	with open('housing.csv', 'wb') as g:
		g.write( re.sub( ' +', ',' ,re.sub( '^ +', '', re.sub('\n +', '\n', f.read()))))