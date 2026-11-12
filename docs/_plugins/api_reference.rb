# frozen_string_literal: true

# Generate reference pages without importing the optional Python dependencies.
require "json"
require "open3"

class OrbitoptApiReference < Jekyll::Generator
  priority :high

  def generate(site)
    script = File.expand_path("../../scripts/generate_api_reference.py", __dir__)
    output, error, status = Open3.capture3("python3", script)
    unless status.success?
      raise Jekyll::Errors::FatalException, "API reference generation failed: #{error}"
    end

    JSON.parse(output).each do |spec|
      directory = spec.fetch("url").sub(%r{\A/}, "")
      page = Jekyll::PageWithoutAFile.new(site, site.source, directory, "index.md")
      page.data.merge!(
        "layout" => "default",
        "title" => spec.fetch("title"),
        "permalink" => spec.fetch("url"),
        "api_generated" => true,
        "nav_exclude" => true,
        "search_exclude" => false
      )
      page.content = spec.fetch("content").gsub("{BASE}", site.baseurl.to_s)
      site.pages << page
    end
  end
end
