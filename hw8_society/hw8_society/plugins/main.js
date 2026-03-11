// Include this script to include everything you need.

(function() {
  const head = document.getElementsByTagName('head')[0];

  function includeScript(src, text) {
    const script = document.createElement('script');
    script.src = src;
    if (text) script.text = text;
    head.appendChild(script);
    return script;
  }

  function includeStylesheet(href) {
    const css = document.createElement('link');
    css.setAttribute('rel', 'stylesheet');
    css.setAttribute('href', href);
    head.appendChild(css);
    return css;
  }

  function initMathJax(scriptLocation, fallbackScriptLocation) {
    if (typeof(sfig) != 'undefined') return;  // Already loaded through sfig
    const script = includeScript(scriptLocation);
    let buf = '';
    buf += 'MathJax.Hub.Config({';
    buf += '  extensions: ["tex2jax.js", "TeX/AMSmath.js", "TeX/AMSsymbols.js"],';
    buf += '  tex2jax: {inlineMath: [["$", "$"]]},';
    buf += '});';
    script.innerHTML = buf;

    // If fail, try the fallback location
    script.onerror = function() {
      if (fallbackScriptLocation)
        initMathJax(fallbackScriptLocation, null);
    }
  }

  initMathJax('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=default');

  includeScript('plugins/jquery.min.js');
  includeStylesheet('plugins/main.css');
})();

function fixScrollPosition() {
  let store = {};
  if (typeof(localStorage) != 'undefined' && localStorage != null) store = localStorage;

  const scrollTopKey = window.location.pathname+'.scrollTop';
  // Because we insert MathJax, we lose the scrolling position, so we have to
  // put it back manually.
  window.onscroll = function() {
    store[scrollTopKey] = document.body.scrollTop;
  }
  if (store.scrollTop)
    window.scrollTo(0, store[scrollTopKey]);
}

function onLoad(assignmentId, ownerName, version, edLink) {
  // Insert generic text
  const header = $('#assignmentHeader');

  header.append($('<div>')
    .append($('<h1>', {class: 'assignmentTitle'}).append(document.title))
    .append($('<div>').append('Stanford CS221 Autumn 2025'.bold())));
  header.append($('<p>').append('Owner CA: ' + ownerName));
  header.append($('<p>').append('Version: ' + version));
  // header.append($('<a>', {href: edLink}).append('Ed Release Post'));

  header.append('<hr>');
  header.append($('<h2>', {class: 'problemTitle'}).append('General Instructions'));
  header.append($('<p>').append('<b>Learning Goals:</b> This assignment is designed to help you develop a critical understanding of the societal impact of an AI tool of your choice. You’ll examine the product from ethical, economic, and sociocultural perspectives, learning to evaluate it not just as a user but as an informed reviewer who can spot ethical issues, question company policies, and understand how business models shape behavior and accessibility. You’ll practice gathering and synthesizing information from policies, media, and hands-on use while becoming aware of potential harms, disparities, and misuse. Ultimately, the goal is to help you begin to think about how a user might assess transparency, fairness, accountability, and data practices in real AI systems and to imagine more equitable versions of these tools and who needs to be involved in creating them.'));

  header.append($('<p>').append('This assignment is only written. <b>It does not have a programming part.</b>'));

  header.append($('<p>').append("All written answers must be <b>typeset (preferably in LaTeX)</b>. We strongly recommend using Overleaf. A link to a tex file with prompts can be found on Ed and a link to a starter guide and a generic LaTeX written answer template is provided on the main course page."));

  header.append($('<p>').append("Also note that your answers should be <b>in order</b> and <b>clearly and correctly labeled</b> to receive credit. Be sure to submit your final answers as a PDF and tag all pages correctly when submitting to Gradescope."));

  header.append($('<p>').append("<b>Only for this homework:</b> Each part is frontloaded with <b>In this part</b>, <b>Why it's important</b>, and <b>Suggested Resources</b> sections. The resources listed are there to speed up your research process and point you in the right direction for any and all subproblems in that part once you get to them."));

  header.append($('<p>').append("<b>Only for this homework:</b> In this assignment, we will ask you to consider the societal implications of your chosen product along many dimensions. If a particular subproblem doesn't seem to apply directly to your product, for full credit, you can <b>write a best-faith answer based on your knowledge and research, or write about how the dimension(s) in question may affect related products.</b> If this is the case for a subproblem, please say that in your answer."));


  header.append('<hr>');

  // Set up survey/feedback section.
  const feedbackSection = $('#feedback');

  feedbackSection.append('<hr>');

  feedbackSection.append($('<h2>', {class: 'problemTitle'}).append('Weekly Feedback'));

  feedbackSection.append($('<p>').append(
    "Given the remote format of the class, we on the teaching team want to know " +
      "how everyone is feeling about the course week-to-week and what we can do to help. " +
  "To help us do that, we'd very much appreciate if you filled out this optional check-in survey to let us know how you're doing: "));

  feedbackSection.append($('<a>', {href: feedbackSection.data("survey-url"), target: "_blank"}).append('Weekly Feedback Survey'));

  feedbackSection.append($('<p>').append(
    "The survey is completely optional and will not affect your grade. All responses will be completely anonymous and strictly confidential. Thanks!"));

  feedbackSection.append('<hr>');

  // Link to code (any mention of *.py).
  $('code').each(function(i, elem) {
    if (true)  {
      const value = elem.innerHTML;
      if (value.match(/^\S*.py$/))
        elem.innerHTML = '<a href="' + value + '">' + value + '</a>';
    }
  });

  // Render point values
  const maxPoints = {};  // Part to number of maxPoints
  function updatePoints(part) {
    const partName = part['number'].split('-')[0];
    maxPoints[partName] = (maxPoints[partName] || 0) + part['max_points'];
  }
  allResult.tests.forEach(updatePoints);

  function showPoints(i, p) {
    if (!p.attributes.id) {
      console.log("Missing id attribute in", p);
      return;
    }
    const partName = p.attributes.id.value;
    const n = maxPoints[partName];
    const s = '[' + n + ' point' + (n > 1 ? 's' : '') + ']';
    $(p).prepend(s);
  }
  $('.writeup').not('.template').each(showPoints);
  $('.code').not('.template').each(showPoints);

  fixScrollPosition();
}
