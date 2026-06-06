---
title: "Posts by Category"
layout: null
permalink: /categories/
analytics: true
---
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Posts by Category | Mason Yahr</title>
    <meta name="description" content="Browse Mason Yahr's writing by category.">
    <link rel="canonical" href="{{ '/categories/' | absolute_url }}">
    <link rel="stylesheet" href="{{ '/assets/css/home-redesign.css' | relative_url }}">
    {% include analytics.html %}
  </head>
  <body class="home-redesign taxonomy-page">
    <main class="taxonomy-shell">
      <nav class="site-nav" aria-label="Primary">
        <a href="{{ '/' | relative_url }}">Mason Yahr</a>
        <a href="{{ '/posts/' | relative_url }}">Posts</a>
        <a href="{{ '/tags/' | relative_url }}">Tags</a>
        <a href="{{ '/about/' | relative_url }}">About</a>
        <a href="mailto:yahr.mason@gmail.com">Contact</a>
      </nav>

      <header class="taxonomy-header">
        <p class="eyebrow">Archive</p>
        <h1>Posts by category.</h1>
        <p>Technical notes grouped by the broad topic they belong to.</p>
      </header>

      <section class="taxonomy-list" aria-label="Posts by category">
        {% assign categories = site.categories | sort %}
        {% for category in categories %}
          {% assign category_name = category[0] %}
          {% assign posts = category[1] %}
          <article class="taxonomy-group">
            <h2>{{ category_name }}</h2>
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
