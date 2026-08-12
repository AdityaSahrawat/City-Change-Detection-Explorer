import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "maplibre-gl/dist/maplibre-gl.css";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "City Change Detection Explorer",
  description:
    "Geospatial land-cover change detection using Sentinel-2 multispectral imagery. " +
    "Visualise NDVI / NDWI / NDBI differences, quantified in hectares and km².",
  keywords: ["satellite", "Sentinel-2", "NDVI", "change detection", "GIS", "geospatial"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={inter.variable}>
      <body>{children}</body>
    </html>
  );
}
