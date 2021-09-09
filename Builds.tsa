fallback_session = get_option('fallback-session')

session_configuration = configuration_data()
session_configuration.set('FALLBACK_SESSION', fallback_session)

pantheon_components = [
  'gala',
  'gala-daemon',
]

gsd_components = [
  'org.gnome.SettingsDaemon.A11ySettings',
  'org.gnome.SettingsDaemon.Color',
  'org.gnome.SettingsDaemon.Datetime',
  'org.gnome.SettingsDaemon.Housekeeping',
  'org.gnome.SettingsDaemon.Keyboard',
  'org.gnome.SettingsDaemon.MediaKeys',
  'org.gnome.SettingsDaemon.Power',
  'org.gnome.SettingsDaemon.PrintNotifications',
  'org.gnome.SettingsDaemon.Rfkill',
  'org.gnome.SettingsDaemon.Sharing',
  'org.gnome.SettingsDaemon.Smartcard',
  'org.gnome.SettingsDaemon.Sound',
  'org.gnome.SettingsDaemon.Wacom',
  'org.gnome.SettingsDaemon.XSettings',
]

# Removed in 3.33.90
# https://gitlab.gnome.org/GNOME/gnome-session/-/commit/863ff3c8e3a64ef6dc6ecd0c243699b598484b44
# https://gitlab.gnome.org/GNOME/gnome-session/-/commit/01f27ed6e501bfd66a7e725a1ca1b452f52fcf07
gsd_mouse_clipboard = [
  'org.gnome.SettingsDaemon.Clipboard',
  'org.gnome.SettingsDaemon.Mouse',
]

# Added in 3.36.0
# https://gitlab.gnome.org/GNOME/gnome-session/-/commit/86d4132c7810b5c1b2392f63c15b3b90b8a7a4f9
gsd_usb_protection = ['org.gnome.SettingsDaemon.UsbProtection']

# Needed where we're not using light-locker and are using Gala above 3.3.2
# We're using 3.36 to target this as it's the first elementary release where this is true
gsd_screensaver_proxy = ['org.gnome.SettingsDaemon.ScreensaverProxy']

# Needs a minimum 3.28 GNOME stack
gsd_minimum_version = '>= 3.27.90'

gsd_dep = dependency('gnome-settings-daemon', version: gsd_minimum_version)
gsd_version = gsd_dep.version()

# Merge the gsd_components list depending on the gnome-settings-daemon version.
if gsd_version.version_compare('>=3.36.0')
  gsd_components += gsd_usb_protection
  gsd_components += gsd_screensaver_proxy
elif gsd_version.version_compare('< 3.33.90')
  gsd_components += gsd_mouse_clipboard
endif

session_components = pantheon_components + gsd_components

session_configuration.set('SESSION_COMPONENTS', ';'.join(session_components))

pantheon_session = configure_file(
  input: 'pantheon.session.in',
  output: '@BASENAME@',
  configuration: session_configuration,
  install_dir: join_paths(datadir, 'gnome-session', 'sessions')
)

autostartdir = join_paths(get_option('sysconfdir'), 'xdg', 'autostart')

gsd_dir = gsd_dep.get_pkgconfig_variable('prefix')
desktop_file_install = find_program('desktop-file-install')

foreach component : gsd_components
  original_filename = component + '.desktop'

  if get_option('detect-program-prefixes') == true
    original_autostart_file = join_paths(gsd_dir, 'etc/xdg/autostart', original_filename)
  else
    original_autostart_file = join_paths(autostartdir, original_filename)
  endif

  patched_filename = 'pantheon-' + component + '.desktop'

  custom_target('create-gsd-autostart' + patched_filename,
    input: original_autostart_file,
    output: patched_filename,
    command: [
      desktop_file_install,
      '--set-key=OnlyShowIn',
      '--set-value=Pantheon',
      '--vendor=pantheon',
      '--dir=@OUTDIR@',
      '@INPUT@',
    ],
    install: true,
    install_dir: autostartdir,
  )
endforeach

gnome_keyring = find_program('gnome-keyring-daemon')
onboard = find_program('onboard')
orca = find_program('orca')

if get_option('detect-program-prefixes') == true
  # TODO: use fs module in meson 0.53.0
  gnome_keyring_prefix = gnome_keyring.path().split('/bin')[0]
  onboard_prefix = onboard.path().split('/bin')[0]
  orca_prefix = orca.path().split('/bin')[0]

  autostarts = {
    'gnome-keyring-pkcs11': join_paths(gnome_keyring_prefix, 'etc/xdg/autostart', 'gnome-keyring-pkcs11.desktop'),
    'gnome-keyring-secrets': join_paths(gnome_keyring_prefix, 'etc/xdg/autostart', 'gnome-keyring-secrets.desktop'),
    'gnome-keyring-ssh': join_paths(gnome_keyring_prefix, 'etc/xdg/autostart', 'gnome-keyring-ssh.desktop'),
    'onboard-autostart': join_paths(onboard_prefix, 'etc/xdg/autostart', 'onboard-autostart.desktop'),
    'orca-autostart': join_paths(orca_prefix, 'etc/xdg/autostart', 'orca-autostart.desktop'),
  }
else
  autostarts = {
    'gnome-keyring-pkcs11': join_paths(autostartdir, 'gnome-keyring-pkcs11.desktop'),
    'gnome-keyring-secrets': join_paths(autostartdir, 'gnome-keyring-secrets.desktop'),
    'gnome-keyring-ssh': join_paths(autostartdir, 'gnome-keyring-ssh.desktop'),
    'onboard-autostart': join_paths(autostartdir, 'onboard-autostart.desktop'),
    'orca-autostart': join_paths(autostartdir, 'orca-autostart.desktop'),
  }
endif

foreach name, autostart : autostarts
  custom_target('create-autostart-' + name,
    input: autostart,
    output: 'pantheon-' + name + '.desktop',
    command: [
      desktop_file_install,
      '--set-key=OnlyShowIn',
      '--set-value=Pantheon',
      '--vendor=pantheon',
      '--dir=@OUTDIR@',
      '@INPUT@',
    ],
    install: true,
    install_dir: autostartdir,
  )
endforeach
