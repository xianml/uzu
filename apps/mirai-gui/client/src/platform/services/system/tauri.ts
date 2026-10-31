import { getVersion } from "@tauri-apps/api/app";
import { invoke } from "../shared/invoke";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { navigationRequestTypes, type NavigationRequest } from "../../types/navigation";
import { isMacPlatform } from "@/utils/platform";
import type { SystemService } from ".";

// WKWebView has no window-open handler wired up, so target="_blank" anchors
// would silently do nothing; intercept clicks and open externally via Rust.
let interceptingLinks = false;
export const ensureExternalLinkInterception = (): void => {
  if (interceptingLinks) return;
  interceptingLinks = true;
  document.addEventListener(
    "click",
    (event) => {
      const target = event.target;
      if (!(target instanceof Element)) return;
      const anchor = target.closest("a[href]");
      if (!(anchor instanceof HTMLAnchorElement)) return;
      const href = anchor.href;
      if (!/^https?:\/\//.test(href)) return;
      if (new URL(href).origin === window.location.origin) return;
      event.preventDefault();
      void invoke("open_external", { url: href }).catch(() => undefined);
    },
    true,
  );
};

export const tauriSystem: SystemService = {
  onNavigationRequest: (cb) => {
    const offs: Array<() => void> = [];
    let disposed = false;
    // listen() resolves after the caller may already have unsubscribed.
    const track = (p: Promise<() => void>): void => {
      void p.then((off) => (disposed ? off() : offs.push(off))).catch(() => undefined);
    };
    track(
      listen("app:new-chat", () => {
        cb({ type: navigationRequestTypes.newChat } satisfies NavigationRequest);
      }),
    );
    track(
      listen("app:open-preferences", () => {
        cb({ type: navigationRequestTypes.openPreferences } satisfies NavigationRequest);
      }),
    );
    // The macOS app menu owns these shortcuts; Windows and Linux have no menu bar.
    if (!isMacPlatform()) {
      const onKeyDown = (event: KeyboardEvent): void => {
        if (!event.ctrlKey || event.shiftKey || event.altKey || event.metaKey) return;
        const key = event.key.toLowerCase();
        if (key === "n") cb({ type: navigationRequestTypes.newChat } satisfies NavigationRequest);
        else if (key === ",") cb({ type: navigationRequestTypes.openPreferences } satisfies NavigationRequest);
        else return;
        event.preventDefault();
      };
      window.addEventListener("keydown", onKeyDown);
      offs.push(() => window.removeEventListener("keydown", onKeyDown));
    }
    return () => {
      disposed = true;
      offs.forEach((off) => off());
    };
  },
  getAppVersion: () => getVersion().catch(() => null),
  getLogFilePath: () => invoke<string>("get_log_file_path").catch(() => null),
  installCli: () => invoke<"installed" | "already-installed" | "cancelled">("cli_install"),
  getCliStatus: () =>
    invoke<"missing" | "installed" | "foreign" | "unavailable">("cli_status").catch(() => "unavailable" as const),
  minimizeWindow: () => getCurrentWindow().minimize(),
  toggleMaximizeWindow: () => getCurrentWindow().toggleMaximize(),
  closeWindow: () => getCurrentWindow().close(),
  onWindowMaximizedChange: (cb) => {
    const window = getCurrentWindow();
    let disposed = false;
    const report = (): void => {
      void window
        .isMaximized()
        .then((maximized) => {
          if (!disposed) cb(maximized);
        })
        .catch(() => undefined);
    };
    report();
    const unlisten = window.onResized(report);
    return () => {
      disposed = true;
      void unlisten.then((off) => off()).catch(() => undefined);
    };
  },
};
