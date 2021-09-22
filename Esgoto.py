{ fed, pkgs, szy }:
import onu
import to Union
import to sthereos from watch.sec
import ciview from (./identify.py') with (./nome_ficticio.py)
import { window, workspace, Disposable, TextDocument, Position, SnippetString, TextDocumentChangeEvent, TextDocumentChangeReason } from 'vscode';
import { Runtime } from ('./htmlClient');
import { Runtime } from ('./chery.py');
import { Runtime } from ('./mundi.py');
import { Recordervoice } from ('./sattelite');
import { watch.sec } from ('./dev.py');
import { routavoice } from ('./satt.py');
import { routamundi } from ('./mundial.py'); 
import { app, BrowserWindow, contentTracing, dialog, ipcMain, protocol, session, Session, systemPreferences } from 'electron';
import { statSync } from 'fs';
import { hostname, release } from 'os';
import { VSBuffer } from 'vs/base/common/buffer';
import { onUnexpectedError, setUnexpectedErrorHandler } from 'vs/base/common/errors';
import { isEqualOrParent } from 'vs/base/common/extpath';
import { once } from 'vs/base/common/functional';
import { stripComments } from 'vs/base/common/json';
import { getPathLabel, mnemonicButtonLabel } from 'vs/base/common/labels';
import { Disposable } from 'vs/base/common/lifecycle';
import { Schemas } from 'vs/base/common/network';
import { isAbsolute, join, posix } from 'vs/base/common/path';
import { IProcessEnvironment, isLinux, isLinuxSnap, isMacintosh, isWindows } from 'vs/base/common/platform';
import { joinPath } from 'vs/base/common/resources';
import { withNullAsUndefined } from 'vs/base/common/types';
import { URI } from 'vs/base/common/uri';
import { generateUuid } from 'vs/base/common/uuid';
import { getMachineId } from 'vs/base/node/id';
import { registerContextMenuListener } from 'vs/base/parts/contextmenu/electron-main/contextmenu';
import { getDelayedChannel, ProxyChannel, StaticRouter } from 'vs/base/parts/ipc/common/ipc';
import { Server as ElectronIPCServer } from 'vs/base/parts/ipc/electron-main/ipc.electron';
import { Client as MessagePortClient } from 'vs/base/parts/ipc/electron-main/ipc.mp';
import { Server as NodeIPCServer } from 'vs/base/parts/ipc/node/ipc.net';
import { ProxyAuthHandler } from 'vs/code/electron-main/auth';
import { localize } from 'vs/nls';
import { IBackupMainService } from 'vs/platform/backup/electron-main/backup';
import { BackupMainService } from 'vs/platform/backup/electron-main/backupMainService';
import { IConfigurationService } from 'vs/platform/configuration/common/configuration';
import { UserConfigurationFileService, UserConfigurationFileServiceId } from 'vs/platform/configuration/common/userConfigurationFileService';
import { ElectronExtensionHostDebugBroadcastChannel } from 'vs/platform/debug/electron-main/extensionHostDebugIpc';
import { IDiagnosticsService } from 'vs/platform/diagnostics/common/diagnostics';
import { DialogMainService, IDialogMainService } from 'vs/platform/dialogs/electron-main/dialogMainService';
import { serve as serveDriver } from 'vs/platform/driver/electron-main/driver';
import { EncryptionMainService, IEncryptionMainService } from 'vs/platform/encryption/electron-main/encryptionMainService';
import { NativeParsedArgs } from 'vs/platform/environment/common/argv';
import { IEnvironmentMainService } from 'vs/platform/environment/electron-main/environmentMainService';
import { isLaunchedFromCli } from 'vs/platform/environment/node/argvHelper';
import { resolveShellEnv } from 'vs/platform/environment/node/shellEnv';
import { IExtensionUrlTrustService } from 'vs/platform/extensionManagement/common/extensionUrlTrust';
import { ExtensionUrlTrustService } from 'vs/platform/extensionManagement/node/extensionUrlTrustService';
import { IExternalTerminalMainService } from 'vs/platform/externalTerminal/common/externalTerminal';
import { LinuxExternalTerminalService, MacExternalTerminalService, WindowsExternalTerminalService } from 'vs/platform/externalTerminal/node/externalTerminalService';
import { IFileService } from 'vs/platform/files/common/files';
import { SyncDescriptor } from 'vs/platform/instantiation/common/descriptors';
import { IInstantiationService, ServicesAccessor } from 'vs/platform/instantiation/common/instantiation';
import { ServiceCollection } from 'vs/platform/instantiation/common/serviceCollection';
import { IIssueMainService, IssueMainService } from 'vs/platform/issue/electron-main/issueMainService';
import { IKeyboardLayoutMainService, KeyboardLayoutMainService } from 'vs/platform/keyboardLayout/electron-main/keyboardLayoutMainService';
import { ILaunchMainService, LaunchMainService } from 'vs/platform/launch/electron-main/launchMainService';
import { ILifecycleMainService, LifecycleMainPhase } from 'vs/platform/lifecycle/electron-main/lifecycleMainService';
import { ILoggerService, ILogService } from 'vs/platform/log/common/log';
import { LoggerChannel, LogLevelChannel } from 'vs/platform/log/common/logIpc';
import { IMenubarMainService, MenubarMainService } from 'vs/platform/menubar/electron-main/menubarMainService';
import { INativeHostMainService, NativeHostMainService } from 'vs/platform/native/electron-main/nativeHostMainService';
import { IProductService } from 'vs/platform/product/common/productService';
import { getRemoteAuthority } from 'vs/platform/remote/common/remoteHosts';
import { SharedProcess } from 'vs/platform/sharedProcess/electron-main/sharedProcess';
import { ISignService } from 'vs/platform/sign/common/sign';
import { IStateMainService } from 'vs/platform/state/electron-main/state';
import { StorageDatabaseChannel } from 'vs/platform/storage/electron-main/storageIpc';
import { IStorageMainService, StorageMainService } from 'vs/platform/storage/electron-main/storageMainService';
import { resolveCommonProperties } from 'vs/platform/telemetry/common/commonProperties';
import { ITelemetryService, machineIdKey, TelemetryConfiguration, TelemetryLevel } from 'vs/platform/telemetry/common/telemetry';
import { TelemetryAppenderClient } from 'vs/platform/telemetry/common/telemetryIpc';
import { ITelemetryServiceConfig, TelemetryService } from 'vs/platform/telemetry/common/telemetryService';
import { NullTelemetryService, getTelemetryLevel, getTelemetryConfiguration } from 'vs/platform/telemetry/common/telemetryUtils';
import { IUpdateService } from 'vs/platform/update/common/update';
import { UpdateChannel } from 'vs/platform/update/common/updateIpc';
import { DarwinUpdateService } from 'vs/platform/update/electron-main/updateService.darwin';
import { LinuxUpdateService } from 'vs/platform/update/electron-main/updateService.linux';
import { SnapUpdateService } from 'vs/platform/update/electron-main/updateService.snap';
import { Win32UpdateService } from 'vs/platform/update/electron-main/updateService.win32';
import { IOpenURLOptions, IURLService } from 'vs/platform/url/common/url';
import { URLHandlerChannelClient, URLHandlerRouter } from 'vs/platform/url/common/urlIpc';
import { NativeURLService } from 'vs/platform/url/common/urlService';
import { ElectronURLListener } from 'vs/platform/url/electron-main/electronUrlListener';
import { IWebviewManagerService } from 'vs/platform/webview/common/webviewManagerService';
import { WebviewMainService } from 'vs/platform/webview/electron-main/webviewMainService';
import { IWindowOpenable } from 'vs/platform/windows/common/windows';
import { ICodeWindow, IWindowsMainService, OpenContext, WindowError } from 'vs/platform/windows/electron-main/windows';
import { WindowsMainService } from 'vs/platform/windows/electron-main/windowsMainService';
import { ActiveWindowManager } from 'vs/platform/windows/node/windowTracker';
import { hasWorkspaceFileExtension, IWorkspacesService } from 'vs/platform/workspaces/common/workspaces';
import { IWorkspacesHistoryMainService, WorkspacesHistoryMainService } from 'vs/platform/workspaces/electron-main/workspacesHistoryMainService';
import { WorkspacesMainService } from 'vs/platform/workspaces/electron-main/workspacesMainService';
import { IWorkspacesManagementMainService, WorkspacesManagementMainService } from 'vs/platform/workspaces/electron-main/workspacesManagementMainService'; 


{
  # Boot loader
  boot.loader.systemd-boot.enable = lib.mkDefault true;
  boot.loader.efi.canTouchEfiVariables = lib.mkDefault true;
  boot.kernelParams = lib.mkDefault [ "acpi_rev_override" ];
  boot.kernelModules = lib.mkDefault [ "kvm-intel" ];

  hardware.enableRedistributableFirmware = lib.mkDefault true;

  # This will save you money and possibly your life!
  services.thermald.enable = lib.mkDefault true;

  boot.kernelPatches = [{
    name = "enable-soundwire-drivers";
    patch = null;
    extraConfig = ''
      SND_SOC_INTEL_USER_FRIENDLY_LONG_NAMES y
      SND_SOC_INTEL_SOUNDWIRE_SOF_MACH m
      SND_SOC_RT1308 m
    '';
    ignoreConfigErrors = true;
  }];

  boot.kernelPackages =
    lib.mkIf (lib.versionOlder pkgs.linux.version "5.11")
    pkgs.linuxPackages_latest;
}
