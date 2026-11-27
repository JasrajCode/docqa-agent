"""Dark theme for the Swagger UI page, injected into FastAPI's default /docs HTML."""

DARK_THEME_CSS = """
body { background: #1b1b1b; margin: 0; }
.swagger-ui { color: #e6e6e6; }
.swagger-ui .info .title, .swagger-ui .info h1, .swagger-ui .info h2, .swagger-ui .info h3,
.swagger-ui .info p, .swagger-ui .info li, .swagger-ui .info table,
.swagger-ui .renderedMarkdown p { color: #c9ccd1; }
.swagger-ui .info .title { color: #f2f2f2; }
.swagger-ui .info a, .swagger-ui .info .base-url { color: #7db8ff; }
.swagger-ui .info blockquote {
  margin: 12px 0; padding: 10px 14px; border-left: 4px solid #e0a63a;
  background: rgba(224, 166, 58, 0.08); border-radius: 4px;
}
.swagger-ui .info blockquote p { margin: 0; color: #e8d3a8; }
.swagger-ui .info code, .swagger-ui .renderedMarkdown code {
  background: #2a2a2a; color: #f0f0f0; padding: 1px 5px; border-radius: 3px;
}

.swagger-ui .opblock-tag { color: #f2f2f2; border-bottom-color: #3a3a3a; }
.swagger-ui .opblock-tag:hover { background: rgba(255, 255, 255, 0.03); }
.swagger-ui .opblock .opblock-summary-path,
.swagger-ui .opblock .opblock-summary-path a { color: #f2f2f2; }
.swagger-ui .opblock .opblock-summary-description { color: #a9adb3; }
.swagger-ui .opblock.opblock-get { background: #1c2230; border-color: #61affe; }
.swagger-ui .opblock.opblock-post { background: #1b2a22; border-color: #49cc90; }
.swagger-ui .opblock .opblock-section-header {
  background: #232323; box-shadow: none; border-bottom: 1px solid #333;
}
.swagger-ui .opblock .opblock-section-header h4,
.swagger-ui .opblock .opblock-section-header label,
.swagger-ui .opblock-description-wrapper p, .swagger-ui .opblock-external-docs-wrapper p,
.swagger-ui .opblock-title_normal p, .swagger-ui .responses-inner h4,
.swagger-ui .responses-inner h5, .swagger-ui .response-col_status,
.swagger-ui .response-col_links, .swagger-ui .response-col_description,
.swagger-ui .parameter__name, .swagger-ui .parameter__type, .swagger-ui .parameter__in,
.swagger-ui .parameter__extension, .swagger-ui .tab li, .swagger-ui .tab li button.tablinks,
.swagger-ui table thead tr th, .swagger-ui table thead tr td,
.swagger-ui .col_header, .swagger-ui .request-url, .swagger-ui .opblock-body h4 { color: #d6d8db; }
.swagger-ui table thead tr th, .swagger-ui table thead tr td { border-bottom-color: #3a3a3a; }
.swagger-ui .response-control-media-type__title { color: #a9adb3; }
.swagger-ui svg.arrow, .swagger-ui .opblock-control-arrow svg,
.swagger-ui .models-control svg, .swagger-ui .opblock-tag svg,
.swagger-ui .model-toggle::after { fill: #c9ccd1; }
.swagger-ui .opblock-summary-control:focus { outline: none; }

.swagger-ui input[type="text"], .swagger-ui textarea, .swagger-ui select {
  background: #262626; color: #e6e6e6; border-color: #4a4a4a;
}
.swagger-ui select { background-image: none; }
.swagger-ui .btn { color: #e6e6e6; border-color: #5a5a5a; background: transparent; }
.swagger-ui .btn.execute { background: #4990e2; border-color: #4990e2; color: #fff; }
.swagger-ui .btn.cancel { color: #ff6b6b; border-color: #ff6b6b; }
.swagger-ui .btn-clear { color: #e6e6e6; }

.swagger-ui section.models { border-color: #3a3a3a; }
.swagger-ui section.models h4, .swagger-ui section.models h4 span { color: #f2f2f2; }
.swagger-ui section.models.is-open h4 { border-bottom-color: #3a3a3a; }
.swagger-ui section.models .model-container,
.swagger-ui section.models .no-margin > .json-schema-2020-12 {
  background: #252525; border-radius: 4px;
}
.swagger-ui section.models .model-container:hover,
.swagger-ui section.models .no-margin > .json-schema-2020-12:hover { background: #2c2c2c; }
.swagger-ui .model, .swagger-ui .model-title, .swagger-ui .model .property,
.swagger-ui .prop-format, .swagger-ui table.model tr td { color: #d6d8db; }
.swagger-ui .json-schema-2020-12__title, .swagger-ui .json-schema-2020-12-head,
.swagger-ui .json-schema-2020-12-property .json-schema-2020-12__title,
.swagger-ui .json-schema-2020-12-keyword__name,
.swagger-ui .json-schema-2020-12-keyword__value { color: #d6d8db; }
.swagger-ui .json-schema-2020-12-expand-deep-button,
.swagger-ui .json-schema-2020-12-accordion { color: #a9adb3; background: transparent; }
.swagger-ui .json-schema-2020-12-accordion__icon svg { fill: #c9ccd1; }
.swagger-ui .json-schema-2020-12__attribute--primary { color: #b392f0; }
.swagger-ui .json-schema-2020-12-body { border-left-color: #3a3a3a; }
.swagger-ui .loading-container .loading::after { color: #c9ccd1; }
"""
