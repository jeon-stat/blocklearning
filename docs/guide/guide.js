// 사용 설명서(guide.html): 명령 복사 단추, 목차에서 지금 읽는 곳 표시
(function () {
  "use strict";
  document.querySelectorAll("pre > code").forEach(function (code) {
    var pre = code.parentNode, b = document.createElement("button");
    b.className = "copy";
    b.type = "button";
    b.textContent = "복사";
    b.onclick = function () {
      var done = function () { b.textContent = "복사됨"; setTimeout(function () { b.textContent = "복사"; }, 1500); };
      if (navigator.clipboard) navigator.clipboard.writeText(code.textContent).then(done, function () {});
    };
    pre.classList.add("has-copy");
    pre.appendChild(b);
  });
  var links = Array.prototype.slice.call(document.querySelectorAll("nav.toc a"));
  if (!("IntersectionObserver" in window) || !links.length) return;
  var seen = {};
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) { seen[e.target.id] = e.isIntersecting; });
    var cur = links.find(function (a) { return seen[a.getAttribute("href").slice(1)]; });
    links.forEach(function (a) { a.classList.toggle("on", a === cur); });
  }, { rootMargin: "-10% 0px -60% 0px" });
  document.querySelectorAll("main section[id]").forEach(function (s) { io.observe(s); });
})();
