let currentSlide = 0;
const slides = document.querySelectorAll(".carousel-image");

function showSlides() {
    // Remove active class from all slides
    slides.forEach(slide => slide.classList.remove("active"));

    // Add active class to the current slide
    slides[currentSlide].classList.add("active");

    // Move to the next slide
    currentSlide = (currentSlide + 1) % slides.length;
}

// Set interval for the carousel (change every 3 seconds)
setInterval(showSlides, 3000);
