//
//  jhPanel.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 2018. 10. 15..
//  Copyright © 2018년 hajunho.com. All rights reserved.
//

import UIKit

class jhPanel : jhDraw, jhPanel_p {
    
    var jhEnforcingMode: Bool = false
    var jhPanelID: Int = 0
    
    internal var mContext : CGContext? = nil
    
    //stored property realted with DATAs
    private var mCountOfDatas : Int = 0
    private var mMaxValueOfDatas : CGFloat = 0
    private var mMinvalueOfDatas : CGFloat = 0
    
    internal var mValuesOfDatas : Array<CGFloat> = Array() {
        didSet {
            if GS.shared.logLevel.contains(.graph) {
                print("mValuesOfDatas.count has been changed to \(mValuesOfDatas.count) in jhPanel")
            }
        }
    }
    
    //stored property related with Drawing
    private let mFixedPanelWidth : CGFloat = jhDraw.maxR //basic ratio 0~10000.0
    private let mFixedPanelHeight : CGFloat = jhDraw.maxR  //basic ratio
    
    //    private var mMargin : CGFloat = 1333.3 //1000.0 is 13.3..%, margin between panel & graph area 0<=martgin<10000.0
    internal var mMargin : CGFloat = 300 //1000.0 is 13.3..%, margin between panel & graph area 0<=martgin<10000.0
    
    private var mPanelWidth : CGFloat? = nil
    private var mPanelHeight : CGFloat? = nil
    
    private var mLineWidth : CGFloat = 1
    private var mColor : CGColor = UIColor.blue.cgColor
    
    
    /// Axes
    var mCountOfaxes_view : Int = 1
    var mUnitOfHorizontalAxes : CGFloat = 100
    var mcountOfHorizontalAxes : Int = 3
    
    internal var mVerticalRatioToDraw_view : CGFloat = 1.0
    
    //calculated property related with DATAs' View
    private var mAllofCountOfDatas : Int {
        get {
            return self.mValuesOfDatas.count
        }
    }
    
    var axisDistance : CGFloat {
        get {
            return (jhDraw.maxR  - mMargin * 2) / CGFloat(mCountOfaxes_view+1)
        }
        set(distance) {
            mCountOfaxes_view = Int(jhDraw.maxR  / CGFloat(distance))
        }
    }
    
    override init(frame: CGRect) {
        if GS.shared.logLevel.contains(.graphPanel) { print("jhPanel override init(\(frame.width), \(frame.height))")}
        super.init(frame: frame)
        self.layer.isGeometryFlipped = true
        mContext = UIGraphicsGetCurrentContext()
        self.mPanelWidth = frame.width
        self.mPanelHeight = frame.height
        if GS.shared.logLevel.contains(.graphPanel) { print("jhPanel init color", mLineWidth)}
    }
    
    override func draw(_ rect: CGRect) {
        if GS.shared.logLevel.contains(.graphPanel) {
            print("jhPanel draw()")
        }
        
        self.mContext = UIGraphicsGetCurrentContext()
        
        drawPanel()
    }
    
    func drawPanel() {
        if GS.shared.logLevel.contains(.graphPanel) { print("drawPanel()") }
        
        initDatas()
        drawBackboard()
        
        
        
        drawDatas()
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    func changePanelSize(_ x : CGFloat, _ y : CGFloat) {
        self.mPanelWidth = x
        self.mPanelHeight = y
    }
    
    func getX(_ x: CGFloat) -> CGFloat? {
        var retX : CGFloat? = nil
        retX = x * mPanelWidth! / mFixedPanelWidth
        return retX
    }
    
    func getY(_ y: CGFloat) -> CGFloat? {
        var retY : CGFloat? = nil
        retY = y * mPanelHeight! / mFixedPanelHeight
        return retY
    }
    
    func drawLine(_ x1 : CGFloat, _ y1 : CGFloat, _ x2 : CGFloat, _ y2 : CGFloat) {
        if GS.shared.logLevel.contains(.graph) { print("panel_drawLine_\(x1), \(y1), \(x2), \(y2)")}
        worldLine(context: mContext, getX(x1)!, getY(y1)!, getX(x2)!, getY(y2)!, mLineWidth, mColor)
    }
    
    func drawLineWithColor(_ x1 : CGFloat, _ y1 : CGFloat, _ x2 : CGFloat, _ y2 : CGFloat, lineWidth : CGFloat, color : CGColor) {
        worldLine(context: mContext, getX(x1)!, getY(y1)!, getX(x2)!, getY(y2)!, lineWidth, color)
    }
    
    func drawRect(margin : CGFloat) {
        drawLine(margin, margin, mFixedPanelWidth-margin, margin)
        drawLine(mFixedPanelWidth-margin, margin, mFixedPanelWidth-margin, mFixedPanelHeight-margin)
        drawLine(mFixedPanelWidth-margin, mFixedPanelHeight-margin, margin, mFixedPanelHeight-margin)
        ////For DEBUG
        //        drawLine(0, 0, mFixedPanelWidth, mFixedPanelHeight)
        //        drawLine(0, mFixedPanelHeight, mFixedPanelWidth, 0)
        drawLine(margin, mFixedPanelHeight-margin, margin, margin)
    }
    
    func drawRect(margin : CGFloat, color : CGColor) {
        mColor = color
        drawRect(margin: margin)
    }
    
    func drawEllipse(_ x : CGFloat, _ y : CGFloat, _ width : CGFloat, _ height : CGFloat, thickness : CGFloat, _ color : CGColor){
        //        worldEllipse(context: mContext, getX(x)!, getY(jhDraw.maxR - y)!, width, height, thickness, color)
        if GS.shared.logLevel.contains(.graph) {
            print("worldEllipse(context: mContext,", getX(x+mMargin)!, getY(jhDraw.maxR-y)!, width, height, thickness, color)
        }
        worldEllipse(context: mContext, getX(x+mMargin)!, getY(y)!, width, height, thickness, color)
    }
    
    /// draw X-axes, Y-axes
    func drawBackboard() {
        mColor = jhColor(red: 229, green: 229, blue: 229)
        
        drawRect(margin: mMargin)
        
        mCountOfaxes_view = mAllofCountOfDatas
        
        drawAxes()
    }
    
//    func drawText(str : String, x : CGFloat, y : CGFloat, width : CGFloat, height : CGFloat) -> UIImageView {
//        let renderer = UIGraphicsImageRenderer(size: CGSize(width: width, height: height))
//        let img = renderer.image { ctx in
//            let paragraphStyle = NSMutableParagraphStyle()
//            paragraphStyle.alignment = .center
//            let attrs = [
//                NSAttributedString.Key.strokeColor : UIColor.black,
//                NSAttributedString.Key.foregroundColor : UIColor.white,
//                NSAttributedString.Key.strokeWidth : -2.0,
//                NSAttributedString.Key.font : UIFont(name: "".font1(), size: width/2) as Any
//                ] as [NSAttributedString.Key : Any]
//
//            let string = str
//            string.draw(with: CGRect(x: 0, y: 0, width: width, height: 10), options: .usesLineFragmentOrigin, attributes: attrs, context: nil)
//        }
//        let imageView : UIImageView = UIImageView(frame: CGRect(x: getX(x)!, y: getY(y)!, width: width, height: height))
//        imageView.image = img
//        return imageView
//    }
    
    func drawAxes() {
        var xlocation : CGFloat = 0
        
        for x in 1..<mCountOfaxes_view+1 {
            xlocation = CGFloat(x) * axisDistance + mMargin
            drawLine(xlocation, mMargin, xlocation, jhDraw.maxR-mMargin)
            
            //TODO: LABEL
//            self.addSubview(drawText(str: String(x), x: xlocation-10, y: jhDraw.maxR-mMargin, width: 10, height: 10))
        }
        
        for x in 1..<mcountOfHorizontalAxes+1 {
            let fx = CGFloat(x)*mUnitOfHorizontalAxes*mVerticalRatioToDraw_view + mMargin
            drawLine(mMargin, fx, jhDraw.maxR-mMargin, fx)
            
            //TODO: LABEL
//            self.addSubview(drawText(str: String(x), x: 100, y: fx, width: 10, height: 10))
        }
        
        //TODO: warning guide line. There's a bug.
        drawLineWithColor(mMargin, 20*mUnitOfHorizontalAxes, jhDraw.maxR-mMargin, 20*mUnitOfHorizontalAxes, lineWidth: 2, color: jhColor(red: 254, green: 191, blue: 4))
        drawLineWithColor(mMargin, 60*mUnitOfHorizontalAxes, jhDraw.maxR-mMargin, 60*mUnitOfHorizontalAxes, lineWidth: 2, color: jhColor(red: 251, green: 83, blue: 96))
    }
    
    func initDatas() {
        let dataSource = getArrayOfData()
        
        var maxValue : CGFloat = 0.0
        var minValue : CGFloat = jhDraw.maxR
        
        for element in dataSource {
            let _element = element as! NSArray
            let vDate = _element[0] as! CFDate
            let vNumber = _element[1] as! CGFloat
            
            if GS.shared.logLevel.contains(.graph) {
                print("datasrc2 \(vDate) \(vNumber)")
            }
            
            if vNumber > maxValue { maxValue = vNumber }
            if vNumber < minValue { minValue = maxValue }
            mValuesOfDatas.append(vNumber)
        }
        
        self.mMaxValueOfDatas = maxValue
        self.mMinvalueOfDatas = minValue
        
        self.mVerticalRatioToDraw_view = (jhDraw.maxR - (2*mMargin)) / self.mMaxValueOfDatas
        if GS.shared.logLevel.contains(.graph) {
            print("mVerticalRatioToDraw_view =", mVerticalRatioToDraw_view)
        }
    }
    
    func jhReSize(size : CGSize) {
        //        self.jhSceneFrameHeight = size.width
        //        self.jhSceneFrameHeight = size.height
    }
    
    //This will be moved to jhScene
    func getArrayOfData() -> NSArray {
        return jhFile.legacyConverterToArray("testdata", "plist")!
    }
    
    func drawDatas() {
        //        worldEllipse(context: mContext, 100, 100, 100, 100, 2, UIColor.blue.cgColor)
        
        var pointCloud = Array<CGPoint>()
        var fx, fy : CGFloat
        
        var x : Int = 0
        for y in mValuesOfDatas {
            //ref:drawLine(CGFloat(x)*axisDistance + mMargin, mMargin, CGFloat(x) * axisDistance + mMargin, 10000-mMargin)
            x += 1
            fx = CGFloat(x)*axisDistance
            fy = CGFloat(y)*mVerticalRatioToDraw_view + mMargin
            drawEllipse(fx, fy, 2, 2, thickness: 2, UIColor.blue.cgColor)
            pointCloud.append(CGPoint.init(x: getX(fx+mMargin)!, y: getY(fy)!))
        }
        
        mContext?.move(to: CGPoint.init(x: 0, y: 0))
        mContext?.setStrokeColorSpace(CGColorSpaceCreateDeviceRGB())
        mContext?.setStrokeColor(UIColor.blue.cgColor)
        mContext?.setLineWidth(1.0)
        mContext?.addLines(between: pointCloud)
        mContext?.strokePath()
    }
}
