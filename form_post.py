import requests
from lxml.html import fromstring

rep = requests.get("http://example.webscraping.com/places/default/user/login")
tree = fromstring(rep.text)
data = {}
for e in tree.cssselect("form input"):
    if e.get('name'):
        data[e.get('name')] = e.get('value')
data['email'] = 'ZW282615SC@gmail.com'
data['password'] = '1234SC.ZW'

print(data)
print(rep.cookies)


post = requests.post("http://example.webscraping.com/places/default/user/login", data=data, cookies=rep.cookies)
print(post.url)
