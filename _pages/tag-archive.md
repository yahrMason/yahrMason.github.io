---
title: "Posts by Tag"
permalink: /tags/
layout: null
analytics: true
---
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Posts by Tag | Mason Yahr</title>
    <meta name="description" content="Browse Mason Yahr's writing by tag.">
    <link rel="canonical" href="{{ '/tags/' | absolute_url }}">
    <link rel="stylesheet" href="{{ '/assets/css/home-redesign.css' | relative_url }}">
    {% include analytics.html %}
  </head>
  <body class="home-redesign taxonomy-page">
    <main class="taxonomy-shell">
      <nav class="site-nav" aria-label="Primary">
        <a href="{{ '/' | relative_url }}">Mason Yahr</a>
        <a href="{{ '/posts/' | relative_url }}">Posts</a>
        <a href="{{ '/categories/' | relative_url }}">Categories</a>
        <a href="{{ '/about/' | relative_url }}">About</a>
        <a href="mailto:yahr.mason@gmail.com">Contact</a>
      </nav>

      <header class="taxonomy-header">
        <p class="eyebrow">Archive</p>
        <h1>Posts by tag.</h1>
        <p>Technical notes grouped by the narrower ideas and tools they touch.</p>
      </header>

      <section class="taxonomy-list" aria-label="Posts by tag">
        {% assign tags = site.tags | sort %}
        {% for tag in tags %}
          {% assign tag_name = tag[0] %}
          {% assign posts = tag[1] %}
          <article class="taxonomy-group">
            <h2>{{ tag_name }}</h2>
            <div class="taxonomy-posts">
              {% for post in posts %}
                <a href="{{ post.url | relative_url }}">
                  <strong>{{ post.title }}</strong>
                  <span>{{ post.date | date: "%B %-d, %Y" }}</span>
                </a>
              {% endfor %}
            </div>
          </article>
        {% endfor %}
      </section>
    </main>

    <footer class="site-footer">
      <span>Mason Yahr</span>
      <a href="{{ '/' | relative_url }}">Home</a>
      <a href="{{ '/posts/' | relative_url }}">All posts</a>
      <a href="{{ '/about/' | relative_url }}">About</a>
      <a href="mailto:yahr.mason@gmail.com">yahr.mason@gmail.com</a>
    </footer>
  </body>
</html>
