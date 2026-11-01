export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#121923',
        paper: '#E4E8EE',
        rule: '#2E3A48',
        flagged: '#C4524A',
        verified: '#3F8F82',
        pending: '#A98B55',
        muted: '#8A96A3',
      },
      fontFamily: {
        serif: ['"Source Serif 4"', 'Georgia', 'serif'],
        sans: ['"IBM Plex Sans"', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
 