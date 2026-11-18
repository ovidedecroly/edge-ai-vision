/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#F6F8FB',
        surface: '#FFFFFF',
        primary: {
          DEFAULT: '#173B65',
          hover: '#112C4C',
          light: '#245288'
        },
        secondary: {
          DEFAULT: '#147D83',
          hover: '#0E5C61'
        },
        success: '#238A57',
        warning: '#D58A16',
        danger: '#C94141',
        border: '#DDE3EA',
        main: '#17202A',
        muted: '#667085'
      },
      borderRadius: {
        DEFAULT: '8px',
        md: '10px'
      },
      boxShadow: {
        thin: '0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px 0 rgba(0, 0, 0, 0.03)',
        card: '0 1px 4px 0 rgba(0, 0, 0, 0.06)'
      }
    },
  },
  plugins: [],
}
