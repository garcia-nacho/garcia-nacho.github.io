---
layout: post
title:  "Welcome to my blog!"
---


Finally I have decided to write a blog where I can share all my code and ideas about ML and AI. 
Stay tuned \!

~~~ python
# a comment
import datetime

def get_or_create_user(session, model, **kwargs):
    instance = session.query(model).filter_by(twitter_user_id=kwargs["twitter_user_id"]).first()
	return instance
~~~

![Fig1](/images/Fig1Post1.png)
