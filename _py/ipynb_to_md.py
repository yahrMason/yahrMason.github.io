import os

NOTEBOOK = "2023-06-14-ode-numeric-approx"

if __name__ == "__main__":
    os.system(f"jupyter nbconvert --to markdown {NOTEBOOK}.ipynb")
    os.replace(f"{NOTEBOOK}.md", f"../_posts/{NOTEBOOK}.md")
