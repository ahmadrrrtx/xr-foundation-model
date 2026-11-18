// Progressive enhancement only: the site is fully usable without JavaScript.
(function () {
  'use strict';
  var toggle = document.getElementById('nav-toggle');
  if (!toggle) return;
  var links = document.querySelectorAll('.site-nav a');
  links.forEach(function (link) {
    link.addEventListener('click', function () {
      toggle.checked = false;
    });
  });
  document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') toggle.checked = false;
  });
}());
