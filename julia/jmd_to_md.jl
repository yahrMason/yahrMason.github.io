## Julia Mardown File Path
file_name = "2023-09-16-mcmc-gif"

home_dir = homedir()
repo_dir = "$home_dir\\Documents\\GitHub\\yahrMason.github.io"
cd(repo_dir)

## Weave MD File
using Weave
weave(
    normpath(pwd(), "julia/$file_name/$file_name.jmd"), 
    out_path = :doc, 
    informat="markdown", 
    doctype = "pandoc"
)

## Move Figures
figure_src = "julia/$file_name/figures"
figure_dst = "assets/$(file_name)_files"
if !isdir(figure_dst)
    mkdir(figure_dst)
end
for figure in readdir(figure_src)
    mv(
        normpath(pwd(), "$figure_src/$figure"),
        normpath(pwd(), "$figure_dst/$figure"),
        force = true
    )
end
rm(figure_src)

# Adjust Figure MD path
md_str = read("julia/$file_name/$file_name.md", String)
md_str = replace(md_str, "![](figures/" => "![png](/assets/$(file_name)_files/")
md_str = replace(md_str, ")\\" => ")")
md_str = replace(md_str, "~~~~{.julia}" => "{% highlight julia %}")
md_str = replace(md_str, "~~~~~~~~~~~~~" => "{% endhighlight %}")
md_str = replace(md_str, "~~~~" => "")
md_str = replace(md_str, "Plots.AnimatedGif" => "![gif]")
md_str = replace(md_str, "C:\\\\Users\\\\mason\\\\Documents\\\\GitHub\\\\yahrMason.github.io\n\\\\julia\\\\$file_name\\\\figures\\\\" => "/assets/$(file_name)_files/")
open("julia/$file_name/$file_name.md", "w") do file
    write(file, md_str)
end

# Move Markdown file to posts
mv(
    normpath(pwd(), "julia/$file_name/$file_name.md"),
    normpath(pwd(), "_posts/$file_name.md"),
    force=true
)