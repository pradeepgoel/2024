'This macro iterates through a list is the excel sheet and renames all tabs as per the list.
'Wrote this for Network Rail E&S project. Here if sheet name is same with the name in the list, throws an error.

Sub RenameMacro()

        'Declare i as integer and it will iterate through the list.

        Dim i As Integer
      
        i = 1  ' i is for row

        For Each Sheet In Sheets

            'This condition stops the iterators as soon as this encounter a blanck cell.

            If Cells(i, 1) <> "" Then       '1 is for column A, needs to update as per column

                Sheet.Name = Cells(i, 1).Value

                i = i + 1

            End If

        Next Sheet

End Sub
