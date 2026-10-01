' Double-click launcher.
'
' NOTE: the app window (and, on first run / after a login reset, a
' console for typing the Telegram confirmation code) needs to be visible,
' so this intentionally does NOT hide the window (unlike some launchers
' that run everything silently in the background). Once you've logged in
' once, the session is saved and future runs won't need console input,
' but the console still appears (that's how the app is built) -- you can
' just ignore it and use the app window.
Set objShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")
scriptDir = objFSO.GetParentFolderName(WScript.ScriptFullName)
objShell.CurrentDirectory = scriptDir

exePath = scriptDir & "\dist\TelegramDigest\TelegramDigest.exe"
If objFSO.FileExists(exePath) Then
    objShell.Run chr(34) & exePath & chr(34), 1, False
Else
    objShell.Run chr(34) & scriptDir & "\run_gui.bat" & chr(34), 1, False
End If
