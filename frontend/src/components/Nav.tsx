import { Menu } from "lucide-react";

const NAV_LINKS = ["Architecture", "Metrics", "Physics", "Team"];

export default function Nav({ onUpload }: { onUpload: () => void }) {
  return (
    <nav className="fixed top-0 inset-x-0 z-[100] flex items-center justify-between p-4 sm:p-5">
      {/* Left: logo + wordmark */}
      <div className="flex items-center gap-2">
        <svg width="26" height="26" viewBox="0 0 256 256" fill="#ffffff" aria-hidden>
          <path d="M 256 256 L 128 256 L 0 128 L 128 128 Z M 256 128 L 128 128 L 0 0 L 128 0 Z" />
        </svg>
        <span className="text-white text-2xl font-playfair italic">ChaturVyuha</span>
      </div>

      {/* Center pill */}
      <div className="hidden md:flex absolute left-1/2 -translate-x-1/2 bg-white/20 backdrop-blur-md border border-white/30 rounded-full px-2 py-2 items-center gap-1">
        <button className="bg-white text-gray-900 px-4 py-1.5 rounded-full text-sm font-medium">
          Demo
        </button>
        {NAV_LINKS.map((label) => (
          <button
            key={label}
            className="text-white/80 hover:bg-white/20 hover:text-white px-4 py-1.5 rounded-full text-sm font-medium transition-colors"
          >
            {label}
          </button>
        ))}
      </div>

      {/* Right: upload (desktop) + hamburger (mobile) */}
      <button
        onClick={onUpload}
        className="hidden md:block bg-white text-gray-900 text-sm font-semibold px-6 py-2.5 rounded-full hover:bg-gray-100 transition-colors"
      >
        Upload TIR
      </button>
      <button
        onClick={onUpload}
        aria-label="Menu"
        className="md:hidden text-white"
      >
        <Menu size={24} />
      </button>
    </nav>
  );
}
