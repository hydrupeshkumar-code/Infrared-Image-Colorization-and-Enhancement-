// Bundled placeholder previews so the hero slider looks alive before any job
// runs. Both are inline SVG data-URIs (no network, no asset files needed).

const svg = (inner: string) =>
  "data:image/svg+xml;utf8," +
  encodeURIComponent(
    `<svg xmlns='http://www.w3.org/2000/svg' width='1200' height='800' viewBox='0 0 1200 800'>${inner}</svg>`
  );

// Thermal / inferno-like input (dark -> magenta -> orange -> yellow).
export const PLACEHOLDER_INPUT = svg(`
  <defs>
    <linearGradient id='t' x1='0' y1='1' x2='1' y2='0'>
      <stop offset='0' stop-color='#000004'/>
      <stop offset='0.35' stop-color='#420a68'/>
      <stop offset='0.6' stop-color='#932667'/>
      <stop offset='0.8' stop-color='#dd513a'/>
      <stop offset='1' stop-color='#fca50a'/>
    </linearGradient>
    <radialGradient id='h1' cx='0.3' cy='0.4' r='0.4'>
      <stop offset='0' stop-color='#fcffa4' stop-opacity='0.9'/>
      <stop offset='1' stop-color='#fcffa4' stop-opacity='0'/>
    </radialGradient>
    <radialGradient id='h2' cx='0.72' cy='0.66' r='0.34'>
      <stop offset='0' stop-color='#f98e09' stop-opacity='0.8'/>
      <stop offset='1' stop-color='#f98e09' stop-opacity='0'/>
    </radialGradient>
  </defs>
  <rect width='1200' height='800' fill='url(#t)'/>
  <rect width='1200' height='800' fill='url(#h1)'/>
  <rect width='1200' height='800' fill='url(#h2)'/>
`);

// Colorized RGB-like output (terrain: greens, soil, water).
export const PLACEHOLDER_OUTPUT = svg(`
  <defs>
    <linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>
      <stop offset='0' stop-color='#1f3a2e'/>
      <stop offset='0.5' stop-color='#3f6f43'/>
      <stop offset='1' stop-color='#8a7b4f'/>
    </linearGradient>
    <radialGradient id='w' cx='0.25' cy='0.7' r='0.35'>
      <stop offset='0' stop-color='#2a5f8f' stop-opacity='0.95'/>
      <stop offset='1' stop-color='#2a5f8f' stop-opacity='0'/>
    </radialGradient>
    <radialGradient id='v' cx='0.7' cy='0.3' r='0.4'>
      <stop offset='0' stop-color='#6fae54' stop-opacity='0.85'/>
      <stop offset='1' stop-color='#6fae54' stop-opacity='0'/>
    </radialGradient>
  </defs>
  <rect width='1200' height='800' fill='url(#g)'/>
  <rect width='1200' height='800' fill='url(#w)'/>
  <rect width='1200' height='800' fill='url(#v)'/>
`);
