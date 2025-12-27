window.addEventListener('pageshow', function() {
    const anchor = window.location.hash;
    if (anchor) {
        const el = document.getElementById(anchor.replace(/^#/, ""));
        if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }
});

