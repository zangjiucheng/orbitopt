# frozen_string_literal: true

# Relative links in the docs that leave the site -- ../README.md,
# ../THIRD_PARTY.md, a folder with no page such as gpu-regression/runs/ --
# work when the Markdown is read on GitHub but 404 on the site. After
# rendering, point each one at the same path on GitHub instead (blob/ for a
# file, tree/ for a folder). Links to anything the site does serve are left
# alone.
require "pathname"
require "cgi"

module RepoLinks
  REPO = "https://github.com/zangjiucheng/orbitopt"
  BRANCH = "main"

  module_function

  def served_paths(site)
    @served ||= {}
    @served[site.object_id] ||= begin
      paths = site.pages.map { |p| p.url } + site.static_files.map { |f| f.url }
      paths.flat_map { |u| [u, u.sub(%r{index\.html\z}, ""), u.sub(%r{\.html\z}, "")] }.to_set
    end
  end

  def rewrite(doc)
    return unless doc.output && doc.output_ext == ".html"

    site = doc.site
    source = Pathname.new(site.source)
    page_dir = Pathname.new(File.dirname(File.join(site.source, doc.relative_path)))
    served = served_paths(site)
    doc.output = doc.output.gsub(/(href|src)="([^"#:?]+)((?:[?#][^"]*)?)"/) do
      attr, path, rest = Regexp.last_match(1), Regexp.last_match(2), Regexp.last_match(3)
      next Regexp.last_match(0) if path.start_with?("/")

      target = (page_dir + CGI.unescape(path)).cleanpath
      rel = target.relative_path_from(source).to_s
      inside = !rel.start_with?("..")
      url = "/" + rel + (path.end_with?("/") ? "/" : "")
      next Regexp.last_match(0) if inside && (served.include?(url) || served.include?(url.sub(/\.md\z/, ".html")))

      repo_rel = (Pathname.new("docs") + rel).cleanpath.to_s
      kind = target.directory? ? "tree" : "blob"
      %(#{attr}="#{REPO}/#{kind}/#{BRANCH}/#{repo_rel}#{rest}")
    end
  end
end

Jekyll::Hooks.register [:pages, :documents], :post_render do |doc|
  RepoLinks.rewrite(doc)
end
