const revealItems = document.querySelectorAll(
  '.reveal-up, .reveal-left, .reveal-right, .reveal-text, .mask-reveal'
);

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
      }
    });
  },
  { threshold: 0.22 }
);

revealItems.forEach((item, i) => {
  item.style.transitionDelay = `${(i % 5) * 90}ms`;
  observer.observe(item);
});

const parallaxItems = document.querySelectorAll('.parallax, .show-card');
window.addEventListener('scroll', () => {
  const y = window.scrollY;
  parallaxItems.forEach((el) => {
    const speed = Number(el.dataset.speed || 0.06);
    el.style.transform = `translateY(${y * speed}px)`;
  });
});

const magneticItems = document.querySelectorAll('.magnetic');
magneticItems.forEach((item) => {
  item.addEventListener('mousemove', (event) => {
    const rect = item.getBoundingClientRect();
    const x = event.clientX - rect.left - rect.width / 2;
    const y = event.clientY - rect.top - rect.height / 2;

    item.style.transform = `translate(${x * 0.18}px, ${y * 0.18}px)`;
  });

  item.addEventListener('mouseleave', () => {
    item.style.transform = 'translate(0, 0)';
  });
});

const ctaButton = document.querySelector('.cta .btn-primary');
if (ctaButton) {
  ctaButton.addEventListener('click', () => {
    ctaButton.animate(
      [
        { boxShadow: '0 0 0 rgba(184, 154, 98, 0)' },
        { boxShadow: '0 0 24px rgba(184, 154, 98, 0.65)' },
        { boxShadow: '0 0 0 rgba(184, 154, 98, 0)' }
      ],
      { duration: 900, easing: 'ease-out' }
    );
  });
}
