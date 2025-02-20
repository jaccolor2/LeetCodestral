import Link from 'next/link'

export default function Navigation() {
  return (
    <nav className="p-4 bg-slate-800 text-white">
      <ul className="flex gap-4">
        <li>
          <Link href="/">Home</Link>
        </li>
        <li>
          <Link href="/problems">Problems</Link>
        </li>
        <li>
          <Link href="/auth/login">Login</Link>
        </li>
      </ul>
    </nav>
  )
}